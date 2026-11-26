using System;
using System.IO;

namespace HKRLEnvMod.Action
{
    // Scaled deep residual + multi-layer continuous GRU neural actor.
    // Supports:
    //   - V1 (Magic 0x3E4C5149): 2-lag history (0.16s, 0.48s) + Max/Sum entity MLPs + Pre-LN ResFFN + Residual GRU.
    //   - V2 (Magic 0x3F4C5149): 4-lag history (0.08s, 0.16s, 0.32s, 0.48s) + 6-Token Multi-Head Entity Self-Attention
    //     (Hero + 3 Boss + 2 Disc) + Pre-LN ResFFN + Residual GRU.
    // Uses the exact same causal 192-dim SistersTimingState; zero recollection required.
    internal sealed class SistersScaledRecurrent
    {
        internal const int Magic = 0x3E4C5149;
        internal const int MagicV2 = 0x3F4C5149;
        private const int ObsWidth = SistersTimingState.Width;
        private const int HistoryCapacity = 64;
        private const int NumTokens = 6;
        private static readonly double[] LagsV1 = { .16, .48 };
        private static readonly double[] LagsV2 = { .08, .16, .32, .48 };

        private readonly bool _isV2;
        private readonly double[] _lags;
        private readonly int _dBoss, _dDisc, _dPast, _dTok, _nHeads, _dModel, _dFfn, _nBlocks, _dGru, _nGruLayers, _nActions;
        private readonly int _encInWidth;

        internal PrimitiveInput[] Actions { get; }
        internal float[] Scores { get; }
        internal float[] Embedding { get; }

        // Motion history ring buffer (exact continuity semantics of SistersMotionHistory)
        private readonly float[] _histStates = new float[HistoryCapacity * ObsWidth];
        private readonly float[] _histTimes = new float[HistoryCapacity];
        private int _histNext, _histCount;

        // Entity MLPs: V1 Boss(36->dBoss->dBoss), Disc(14->dDisc->dDisc); V2 Boss(36->dBoss->dTok), Disc(14->dDisc->dTok)
        private readonly float[] _boss1W, _boss1B, _boss2W, _boss2B;
        private readonly float[] _disc1W, _disc1B, _disc2W, _disc2B;
        private readonly float[] _pastW, _pastB;

        // V2 6-Token Multi-Head Entity Self-Attention weights
        private readonly float[]? _heroTokW, _heroTokB;
        private readonly float[]? _attnLnW, _attnLnB, _attnQkvW, _attnQkvB, _attnOutW, _attnOutB;
        private readonly float[]? _tokFfnLnW, _tokFfnLnB, _tokFfn1W, _tokFfn1B, _tokFfn2W, _tokFfn2B;

        // Stem (encInWidth -> dModel + LN + SiLU)
        private readonly float[] _stemW, _stemB, _stemLnW, _stemLnB;

        // Residual Pre-LN FFN blocks
        private readonly float[][] _blkLnW, _blkLnB, _blkFc1W, _blkFc1B, _blkFc2W, _blkFc2B;

        // Multi-layer Residual GRU + Post-LN
        private readonly float[][] _gruWih, _gruWhh, _gruBih, _gruBhh, _gruLnW, _gruLnB;
        private readonly float[][] _h;

        // Folded action head (dModel -> nActions)
        private readonly float[] _headW, _headB;

        // Scratch buffers (zero per-step GC allocations)
        private readonly float[] _encIn, _bossTmp1, _bossTmp2, _discTmp1, _discTmp2;
        private readonly float[] _x, _lnBuf, _ffnBuf, _deltaBuf, _xGates, _hGates, _noMemScores;
        // V2 scratch buffers
        private readonly float[]? _heroIn, _toks, _tokLn, _qkv, _attnCtx, _attnProj, _tokFfnBuf, _tokDelta;
        private readonly float[]? _attnScores;
        private readonly bool[]? _tokValid;

        private float _previous = -1;
        private int _decisions, _changed;

        internal int HiddenFlatLength => _nGruLayers * _dGru;

        internal void CopyHiddenTo(float[] destination)
        {
            for (int l = 0; l < _nGruLayers; l++)
                Array.Copy(_h[l], 0, destination, l * _dGru, _dGru);
        }

        internal void WriteReceipt(string path, ulong episode)
        {
            File.WriteAllText(path, string.Join("\n", episode, _decisions, _changed));
        }

        private SistersScaledRecurrent(BinaryReader reader)
        {
            int magic = reader.ReadInt32();
            if (magic != Magic && magic != MagicV2)
                throw new InvalidDataException("Wrong scaled recurrent magic");
            _isV2 = magic == MagicV2;
            _lags = _isV2 ? LagsV2 : LagsV1;

            int obsWidth = reader.ReadInt32();
            _dBoss = reader.ReadInt32();
            _dDisc = reader.ReadInt32();
            _dPast = reader.ReadInt32();
            if (_isV2)
            {
                _dTok = reader.ReadInt32();
                _nHeads = reader.ReadInt32();
            }
            else
            {
                _dTok = 0;
                _nHeads = 0;
            }
            _dModel = reader.ReadInt32();
            _dFfn = reader.ReadInt32();
            _nBlocks = reader.ReadInt32();
            _dGru = reader.ReadInt32();
            _nGruLayers = reader.ReadInt32();
            _nActions = reader.ReadInt32();

            if (obsWidth != ObsWidth ||
                _dBoss < 1 || _dBoss > 256 ||
                _dDisc < 1 || _dDisc > 256 ||
                _dPast < 1 || _dPast > 256 ||
                _dModel < 1 || _dModel > 1024 ||
                _dFfn < 1 || _dFfn > 2048 ||
                _nBlocks < 0 || _nBlocks > 8 ||
                _dGru != _dModel ||
                _nGruLayers < 1 || _nGruLayers > 4 ||
                _nActions < 1 || _nActions > 4096)
                throw new InvalidDataException("Invalid scaled recurrent architecture header");
            if (_isV2 && (_dTok < 16 || _dTok > 256 || _nHeads < 1 || _nHeads > 8 || (_dTok % _nHeads) != 0))
                throw new InvalidDataException("Invalid V2 entity attention header");

            _encInWidth = _isV2
                ? ObsWidth + _lags.Length * _dPast + 5 * _dTok
                : ObsWidth + 2 * _dBoss + 2 * _dDisc + _lags.Length * _dPast;

            Actions = new PrimitiveInput[_nActions];
            Scores = new float[_nActions];
            _noMemScores = new float[_nActions];
            Embedding = new float[_dModel];

            for (int i = 0; i < _nActions; i++)
            {
                int move = reader.ReadInt32(), aim = reader.ReadInt32(), buttons = reader.ReadInt32();
                if (Math.Abs(move) > 1 || Math.Abs(aim) > 1 || buttons < 0 || buttons > 511)
                    throw new InvalidDataException("Invalid primitive in scaled actor catalogue");
                Actions[i] = new PrimitiveInput(move, aim, (uint)buttons);
            }

            int bossOut = _isV2 ? _dTok : _dBoss;
            int discOut = _isV2 ? _dTok : _dDisc;

            _boss1W = NativeSistersNetwork.ReadWeights(reader, _dBoss * 36);
            _boss1B = NativeSistersNetwork.ReadWeights(reader, _dBoss);
            _boss2W = NativeSistersNetwork.ReadWeights(reader, bossOut * _dBoss);
            _boss2B = NativeSistersNetwork.ReadWeights(reader, bossOut);

            _disc1W = NativeSistersNetwork.ReadWeights(reader, _dDisc * 14);
            _disc1B = NativeSistersNetwork.ReadWeights(reader, _dDisc);
            _disc2W = NativeSistersNetwork.ReadWeights(reader, discOut * _dDisc);
            _disc2B = NativeSistersNetwork.ReadWeights(reader, discOut);

            _pastW = NativeSistersNetwork.ReadWeights(reader, _dPast * (ObsWidth + 1));
            _pastB = NativeSistersNetwork.ReadWeights(reader, _dPast);

            if (_isV2)
            {
                int heroInLen = 24 + 32 + _lags.Length * _dPast;
                _heroTokW = NativeSistersNetwork.ReadWeights(reader, _dTok * heroInLen);
                _heroTokB = NativeSistersNetwork.ReadWeights(reader, _dTok);
                _attnLnW = NativeSistersNetwork.ReadWeights(reader, _dTok);
                _attnLnB = NativeSistersNetwork.ReadWeights(reader, _dTok);
                _attnQkvW = NativeSistersNetwork.ReadWeights(reader, (3 * _dTok) * _dTok);
                _attnQkvB = NativeSistersNetwork.ReadWeights(reader, 3 * _dTok);
                _attnOutW = NativeSistersNetwork.ReadWeights(reader, _dTok * _dTok);
                _attnOutB = NativeSistersNetwork.ReadWeights(reader, _dTok);
                _tokFfnLnW = NativeSistersNetwork.ReadWeights(reader, _dTok);
                _tokFfnLnB = NativeSistersNetwork.ReadWeights(reader, _dTok);
                _tokFfn1W = NativeSistersNetwork.ReadWeights(reader, (2 * _dTok) * _dTok);
                _tokFfn1B = NativeSistersNetwork.ReadWeights(reader, 2 * _dTok);
                _tokFfn2W = NativeSistersNetwork.ReadWeights(reader, _dTok * (2 * _dTok));
                _tokFfn2B = NativeSistersNetwork.ReadWeights(reader, _dTok);

                _heroIn = new float[heroInLen];
                _toks = new float[NumTokens * _dTok];
                _tokLn = new float[_dTok];
                _qkv = new float[NumTokens * 3 * _dTok];
                _attnCtx = new float[_dTok];
                _attnProj = new float[_dTok];
                _tokFfnBuf = new float[2 * _dTok];
                _tokDelta = new float[_dTok];
                _attnScores = new float[NumTokens];
                _tokValid = new bool[NumTokens];
            }

            _stemW = NativeSistersNetwork.ReadWeights(reader, _dModel * _encInWidth);
            _stemB = NativeSistersNetwork.ReadWeights(reader, _dModel);
            _stemLnW = NativeSistersNetwork.ReadWeights(reader, _dModel);
            _stemLnB = NativeSistersNetwork.ReadWeights(reader, _dModel);

            _blkLnW = new float[_nBlocks][];
            _blkLnB = new float[_nBlocks][];
            _blkFc1W = new float[_nBlocks][];
            _blkFc1B = new float[_nBlocks][];
            _blkFc2W = new float[_nBlocks][];
            _blkFc2B = new float[_nBlocks][];
            for (int b = 0; b < _nBlocks; b++)
            {
                _blkLnW[b] = NativeSistersNetwork.ReadWeights(reader, _dModel);
                _blkLnB[b] = NativeSistersNetwork.ReadWeights(reader, _dModel);
                _blkFc1W[b] = NativeSistersNetwork.ReadWeights(reader, _dFfn * _dModel);
                _blkFc1B[b] = NativeSistersNetwork.ReadWeights(reader, _dFfn);
                _blkFc2W[b] = NativeSistersNetwork.ReadWeights(reader, _dModel * _dFfn);
                _blkFc2B[b] = NativeSistersNetwork.ReadWeights(reader, _dModel);
            }

            _gruWih = new float[_nGruLayers][];
            _gruWhh = new float[_nGruLayers][];
            _gruBih = new float[_nGruLayers][];
            _gruBhh = new float[_nGruLayers][];
            _gruLnW = new float[_nGruLayers][];
            _gruLnB = new float[_nGruLayers][];
            _h = new float[_nGruLayers][];
            int gates = 3 * _dGru;
            for (int l = 0; l < _nGruLayers; l++)
            {
                _gruWih[l] = NativeSistersNetwork.ReadWeights(reader, gates * _dGru);
                _gruWhh[l] = NativeSistersNetwork.ReadWeights(reader, gates * _dGru);
                _gruBih[l] = NativeSistersNetwork.ReadWeights(reader, gates);
                _gruBhh[l] = NativeSistersNetwork.ReadWeights(reader, gates);
                _gruLnW[l] = NativeSistersNetwork.ReadWeights(reader, _dGru);
                _gruLnB[l] = NativeSistersNetwork.ReadWeights(reader, _dGru);
                _h[l] = new float[_dGru];
            }

            _headW = NativeSistersNetwork.ReadWeights(reader, _nActions * _dModel);
            _headB = NativeSistersNetwork.ReadWeights(reader, _nActions);

            if (reader.BaseStream.Position != reader.BaseStream.Length)
                throw new InvalidDataException("Trailing scaled actor bytes");

            _encIn = new float[_encInWidth];
            _bossTmp1 = new float[_dBoss];
            _bossTmp2 = new float[bossOut];
            _discTmp1 = new float[_dDisc];
            _discTmp2 = new float[discOut];
            _x = new float[_dModel];
            _lnBuf = new float[_dModel];
            _ffnBuf = new float[_dFfn];
            _deltaBuf = new float[_dModel];
            _xGates = new float[gates];
            _hGates = new float[gates];
        }

        internal static SistersScaledRecurrent Read(BinaryReader reader) => new SistersScaledRecurrent(reader);

        private void CheckHistoryContinuity(float now)
        {
            if (_histCount == 0) return;
            double elapsed = (double)now - _histTimes[(_histNext + HistoryCapacity - 1) % HistoryCapacity];
            if (elapsed < 0 || elapsed > .24001) _histCount = 0;
        }

        internal void Remember(float[] raw, float now)
        {
            CheckHistoryContinuity(now);
            int start = _histNext * ObsWidth;
            Array.Copy(raw, 0, _histStates, start, ObsWidth);
            for (int slot = 0; slot < 5; slot++)
            {
                int k = slot < 3 ? 24 + slot * 36 : 132 + (slot - 3) * 14;
                if (raw[k] == 0) Array.Clear(_histStates, start + k, slot < 3 ? 36 : 14);
            }
            _histTimes[_histNext] = now;
            _histNext = (_histNext + 1) % HistoryCapacity;
            _histCount = Math.Min(HistoryCapacity, _histCount + 1);
        }

        private void EncodeHistory(float now, float[] dst, int offset)
        {
            CheckHistoryContinuity(now);
            Array.Clear(dst, offset, _dPast * _lags.Length);
            foreach (double lag in _lags)
            {
                for (int past = 1; past <= _histCount; past++)
                {
                    int index = (_histNext + HistoryCapacity - past) % HistoryCapacity;
                    double age = (double)now - _histTimes[index];
                    if (age < lag - .00001) continue;
                    if (age <= lag + .24001)
                    {
                        int source = index * ObsWidth;
                        float ageF = (float)age;
                        for (int j = 0; j < _dPast; j++)
                        {
                            float value = _pastB[j];
                            int w = j * (ObsWidth + 1);
                            int k = 0;
                            for (; k + 3 < ObsWidth; k += 4)
                                value += _histStates[source + k] * _pastW[w + k]
                                       + _histStates[source + k + 1] * _pastW[w + k + 1]
                                       + _histStates[source + k + 2] * _pastW[w + k + 2]
                                       + _histStates[source + k + 3] * _pastW[w + k + 3];
                            for (; k < ObsWidth; k++)
                                value += _histStates[source + k] * _pastW[w + k];
                            value += ageF * _pastW[w + ObsWidth];
                            dst[offset + j] = value > 0f ? value : 0f;
                        }
                    }
                    break;
                }
                offset += _dPast;
            }
        }

        private void EncodeSpatialV1(float[] raw, float time)
        {
            Array.Copy(raw, 0, _encIn, 0, ObsWidth);
            int bossMaxOff = ObsWidth;
            int bossSumOff = bossMaxOff + _dBoss;
            int discMaxOff = bossSumOff + _dBoss;
            int discSumOff = discMaxOff + _dDisc;
            int pastOff = discSumOff + _dDisc;
            Array.Clear(_encIn, bossMaxOff, 2 * _dBoss + 2 * _dDisc);

            for (int slot = 0; slot < 3; slot++)
            {
                int k = 24 + slot * 36;
                if (raw[k] == 0f)
                {
                    Array.Clear(_encIn, k, 36);
                    continue;
                }
                DenseSliceSiLU(raw, k, 36, _boss1W, _boss1B, _bossTmp1);
                DenseReLU(_bossTmp1, _boss2W, _boss2B, _bossTmp2);
                for (int j = 0; j < _dBoss; j++)
                {
                    float v = _bossTmp2[j];
                    if (v > _encIn[bossMaxOff + j]) _encIn[bossMaxOff + j] = v;
                    _encIn[bossSumOff + j] += v;
                }
            }

            for (int slot = 0; slot < 2; slot++)
            {
                int k = 132 + slot * 14;
                if (raw[k] == 0f)
                {
                    Array.Clear(_encIn, k, 14);
                    continue;
                }
                DenseSliceSiLU(raw, k, 14, _disc1W, _disc1B, _discTmp1);
                DenseReLU(_discTmp1, _disc2W, _disc2B, _discTmp2);
                for (int j = 0; j < _dDisc; j++)
                {
                    float v = _discTmp2[j];
                    if (v > _encIn[discMaxOff + j]) _encIn[discMaxOff + j] = v;
                    _encIn[discSumOff + j] += v;
                }
            }

            EncodeHistory(time, _encIn, pastOff);
            Remember(raw, time);
        }

        private void EncodeSpatialV2(float[] raw, float time)
        {
            // Layout of _encIn: [raw_clean(192), past(4*dPast), hero_tok(dTok), boss_max(dTok), boss_sum(dTok), disc_max(dTok), disc_sum(dTok)]
            Array.Copy(raw, 0, _encIn, 0, ObsWidth);
            int pastOff = ObsWidth;
            int pastLen = _lags.Length * _dPast;
            EncodeHistory(time, _encIn, pastOff);
            Remember(raw, time);

            float[] toks = _toks!;
            bool[] valid = _tokValid!;
            Array.Clear(toks, 0, toks.Length);

            // Token 0: Hero + Controls + 4-Lag History -> dTok (SiLU)
            valid[0] = true;
            float[] heroIn = _heroIn!;
            Array.Copy(_encIn, 0, heroIn, 0, 24);
            Array.Copy(_encIn, 160, heroIn, 24, 32);
            Array.Copy(_encIn, pastOff, heroIn, 56, pastLen);
            DenseSiLUOffset(heroIn, _heroTokW!, _heroTokB!, toks, 0, _dTok);

            // Tokens 1..3: 3 Bosses
            for (int slot = 0; slot < 3; slot++)
            {
                int k = 24 + slot * 36;
                if (raw[k] == 0f)
                {
                    Array.Clear(_encIn, k, 36);
                    valid[1 + slot] = false;
                    continue;
                }
                valid[1 + slot] = true;
                DenseSliceSiLU(raw, k, 36, _boss1W, _boss1B, _bossTmp1);
                DenseReLU(_bossTmp1, _boss2W, _boss2B, _bossTmp2);
                Array.Copy(_bossTmp2, 0, toks, (1 + slot) * _dTok, _dTok);
            }

            // Tokens 4..5: 2 Discs
            for (int slot = 0; slot < 2; slot++)
            {
                int k = 132 + slot * 14;
                if (raw[k] == 0f)
                {
                    Array.Clear(_encIn, k, 14);
                    valid[4 + slot] = false;
                    continue;
                }
                valid[4 + slot] = true;
                DenseSliceSiLU(raw, k, 14, _disc1W, _disc1B, _discTmp1);
                DenseReLU(_discTmp1, _disc2W, _disc2B, _discTmp2);
                Array.Copy(_discTmp2, 0, toks, (4 + slot) * _dTok, _dTok);
            }

            // Pre-LN Multi-Head Self-Attention across the 6 entity tokens
            float[] qkv = _qkv!;
            float[] tokLn = _tokLn!;
            int qkvStride = 3 * _dTok;
            for (int t = 0; t < NumTokens; t++)
            {
                if (!valid[t]) continue;
                LayerNormOffset(toks, t * _dTok, _dTok, _attnLnW!, _attnLnB!, tokLn);
                DenseOffsetOut(tokLn, _attnQkvW!, _attnQkvB!, qkv, t * qkvStride, qkvStride);
            }

            int headDim = _dTok / _nHeads;
            double scale = 1.0 / Math.Sqrt(headDim);
            float[] attnScores = _attnScores!;
            float[] attnCtx = _attnCtx!;
            float[] attnProj = _attnProj!;

            for (int i = 0; i < NumTokens; i++)
            {
                if (!valid[i]) continue;
                int qBase = i * qkvStride;
                for (int h = 0; h < _nHeads; h++)
                {
                    int hOff = h * headDim;
                    double maxScore = double.NegativeInfinity;
                    for (int j = 0; j < NumTokens; j++)
                    {
                        if (!valid[j]) { attnScores[j] = -1e9f; continue; }
                        int kBase = j * qkvStride + _dTok + hOff;
                        double dot = 0.0;
                        for (int d = 0; d < headDim; d++)
                            dot += qkv[qBase + hOff + d] * qkv[kBase + d];
                        double sc = dot * scale;
                        attnScores[j] = (float)sc;
                        if (sc > maxScore) maxScore = sc;
                    }
                    double sumExp = 0.0;
                    for (int j = 0; j < NumTokens; j++)
                    {
                        if (!valid[j]) { attnScores[j] = 0f; continue; }
                        double w = Math.Exp(attnScores[j] - maxScore);
                        attnScores[j] = (float)w;
                        sumExp += w;
                    }
                    float invSum = (float)(1.0 / sumExp);
                    for (int j = 0; j < NumTokens; j++) attnScores[j] *= invSum;

                    for (int d = 0; d < headDim; d++)
                    {
                        float acc = 0f;
                        for (int j = 0; j < NumTokens; j++)
                        {
                            if (!valid[j]) continue;
                            acc += attnScores[j] * qkv[j * qkvStride + 2 * _dTok + hOff + d];
                        }
                        attnCtx[hOff + d] = acc;
                    }
                }
                Dense(attnCtx, _attnOutW!, _attnOutB!, attnProj);
                int tOff = i * _dTok;
                for (int d = 0; d < _dTok; d++) toks[tOff + d] += attnProj[d];
            }

            // Pre-LN Token FFN (dTok -> 2*dTok SiLU -> dTok) + residual
            float[] tokFfnBuf = _tokFfnBuf!;
            float[] tokDelta = _tokDelta!;
            for (int i = 0; i < NumTokens; i++)
            {
                if (!valid[i]) continue;
                int tOff = i * _dTok;
                LayerNormOffset(toks, tOff, _dTok, _tokFfnLnW!, _tokFfnLnB!, tokLn);
                DenseSiLU(tokLn, _tokFfn1W!, _tokFfn1B!, tokFfnBuf);
                Dense(tokFfnBuf, _tokFfn2W!, _tokFfn2B!, tokDelta);
                for (int d = 0; d < _dTok; d++) toks[tOff + d] += tokDelta[d];
            }

            // Pool into _encIn: [hero_tok, boss_max, boss_sum, disc_max, disc_sum]
            int heroTokOff = pastOff + pastLen;
            int bossMaxOff = heroTokOff + _dTok;
            int bossSumOff = bossMaxOff + _dTok;
            int discMaxOff = bossSumOff + _dTok;
            int discSumOff = discMaxOff + _dTok;

            Array.Copy(toks, 0, _encIn, heroTokOff, _dTok);
            Array.Clear(_encIn, bossMaxOff, 4 * _dTok);

            bool anyBoss = false;
            for (int slot = 0; slot < 3; slot++)
            {
                if (!valid[1 + slot]) continue;
                int src = (1 + slot) * _dTok;
                for (int d = 0; d < _dTok; d++)
                {
                    float v = toks[src + d];
                    if (!anyBoss || v > _encIn[bossMaxOff + d]) _encIn[bossMaxOff + d] = v;
                    _encIn[bossSumOff + d] += v;
                }
                anyBoss = true;
            }

            bool anyDisc = false;
            for (int slot = 0; slot < 2; slot++)
            {
                if (!valid[4 + slot]) continue;
                int src = (4 + slot) * _dTok;
                for (int d = 0; d < _dTok; d++)
                {
                    float v = toks[src + d];
                    if (!anyDisc || v > _encIn[discMaxOff + d]) _encIn[discMaxOff + d] = v;
                    _encIn[discSumOff + d] += v;
                }
                anyDisc = true;
            }
        }

        internal int Choose(float[] raw, float time)
        {
            if (_isV2) EncodeSpatialV2(raw, time);
            else EncodeSpatialV1(raw, time);

            // Stem: Linear -> LayerNorm -> SiLU
            Dense(_encIn, _stemW, _stemB, _lnBuf);
            LayerNormSiLU(_lnBuf, _stemLnW, _stemLnB, _x);

            // Residual Pre-LN FFN Blocks
            for (int b = 0; b < _nBlocks; b++)
            {
                LayerNorm(_x, _blkLnW[b], _blkLnB[b], _lnBuf);
                DenseSiLU(_lnBuf, _blkFc1W[b], _blkFc1B[b], _ffnBuf);
                Dense(_ffnBuf, _blkFc2W[b], _blkFc2B[b], _deltaBuf);
                for (int i = 0; i < _dModel; i++) _x[i] += _deltaBuf[i];
            }

            // Compute no-memory argmax for receipt tracking
            Dense(_x, _headW, _headB, _noMemScores);
            int noMemBest = 0;
            for (int i = 1; i < _nActions; i++)
                if (_noMemScores[i] > _noMemScores[noMemBest]) noMemBest = i;

            // Continuous Multi-Layer Residual GRU
            if (_previous < 0 || time <= _previous || time - _previous > .081f)
            {
                for (int l = 0; l < _nGruLayers; l++) Array.Clear(_h[l], 0, _dGru);
            }
            _previous = time;

            Array.Copy(_x, 0, Embedding, 0, _dModel);
            for (int l = 0; l < _nGruLayers; l++)
            {
                float[] hl = _h[l];
                Dense(Embedding, _gruWih[l], _gruBih[l], _xGates);
                Dense(hl, _gruWhh[l], _gruBhh[l], _hGates);
                for (int i = 0; i < _dGru; i++)
                {
                    float r = Sigmoid(_xGates[i] + _hGates[i]);
                    float z = Sigmoid(_xGates[_dGru + i] + _hGates[_dGru + i]);
                    float n = (float)Math.Tanh(_xGates[2 * _dGru + i] + r * _hGates[2 * _dGru + i]);
                    hl[i] = z * hl[i] + (1f - z) * n;
                    _lnBuf[i] = Embedding[i] + hl[i];
                }
                LayerNorm(_lnBuf, _gruLnW[l], _gruLnB[l], Embedding);
            }

            // Output Head
            Dense(Embedding, _headW, _headB, Scores);
            int best = 0;
            for (int i = 1; i < _nActions; i++)
                if (Scores[i] > Scores[best]) best = i;

            _decisions++;
            if (best != noMemBest) _changed++;
            return best;
        }

        private static float Sigmoid(float x) => (float)(1.0 / (1.0 + Math.Exp(-x)));
        private static float SiLU(float x) => x * (float)(1.0 / (1.0 + Math.Exp(-x)));

        private static void LayerNorm(float[] src, float[] w, float[] b, float[] dst)
        {
            LayerNormOffset(src, 0, src.Length, w, b, dst);
        }

        private static void LayerNormOffset(float[] src, int srcOff, int n, float[] w, float[] b, float[] dst)
        {
            double mean = 0.0;
            for (int i = 0; i < n; i++) mean += src[srcOff + i];
            mean /= n;
            double var = 0.0;
            for (int i = 0; i < n; i++)
            {
                double d = src[srcOff + i] - mean;
                var += d * d;
            }
            double invStd = 1.0 / Math.Sqrt(var / n + 1e-5);
            for (int i = 0; i < n; i++)
                dst[i] = (float)(((src[srcOff + i] - mean) * invStd) * w[i] + b[i]);
        }

        private static void LayerNormSiLU(float[] src, float[] w, float[] b, float[] dst)
        {
            int n = src.Length;
            double mean = 0.0;
            for (int i = 0; i < n; i++) mean += src[i];
            mean /= n;
            double var = 0.0;
            for (int i = 0; i < n; i++)
            {
                double d = src[i] - mean;
                var += d * d;
            }
            double invStd = 1.0 / Math.Sqrt(var / n + 1e-5);
            for (int i = 0; i < n; i++)
            {
                float v = (float)(((src[i] - mean) * invStd) * w[i] + b[i]);
                dst[i] = SiLU(v);
            }
        }

        private static void Dense(float[] x, float[] w, float[] bias, float[] result)
        {
            DenseOffsetOut(x, w, bias, result, 0, result.Length);
        }

        private static void DenseOffsetOut(float[] x, float[] w, float[] bias, float[] result, int outOff, int outLen)
        {
            int inLen = x.Length, k = 0;
            for (int j = 0; j < outLen; j++)
            {
                float s0 = bias[j], s1 = 0f, s2 = 0f, s3 = 0f;
                int i = 0;
                for (; i + 3 < inLen; i += 4, k += 4)
                {
                    s0 += x[i] * w[k];
                    s1 += x[i + 1] * w[k + 1];
                    s2 += x[i + 2] * w[k + 2];
                    s3 += x[i + 3] * w[k + 3];
                }
                float sum = (s0 + s1) + (s2 + s3);
                for (; i < inLen; i++, k++) sum += x[i] * w[k];
                result[outOff + j] = sum;
            }
        }

        private static void DenseReLU(float[] x, float[] w, float[] bias, float[] result)
        {
            int inLen = x.Length, outLen = result.Length, k = 0;
            for (int j = 0; j < outLen; j++)
            {
                float s0 = bias[j], s1 = 0f, s2 = 0f, s3 = 0f;
                int i = 0;
                for (; i + 3 < inLen; i += 4, k += 4)
                {
                    s0 += x[i] * w[k];
                    s1 += x[i + 1] * w[k + 1];
                    s2 += x[i + 2] * w[k + 2];
                    s3 += x[i + 3] * w[k + 3];
                }
                float sum = (s0 + s1) + (s2 + s3);
                for (; i < inLen; i++, k++) sum += x[i] * w[k];
                result[j] = sum > 0f ? sum : 0f;
            }
        }

        private static void DenseSiLU(float[] x, float[] w, float[] bias, float[] result)
        {
            DenseSiLUOffset(x, w, bias, result, 0, result.Length);
        }

        private static void DenseSiLUOffset(float[] x, float[] w, float[] bias, float[] result, int outOff, int outLen)
        {
            int inLen = x.Length, k = 0;
            for (int j = 0; j < outLen; j++)
            {
                float s0 = bias[j], s1 = 0f, s2 = 0f, s3 = 0f;
                int i = 0;
                for (; i + 3 < inLen; i += 4, k += 4)
                {
                    s0 += x[i] * w[k];
                    s1 += x[i + 1] * w[k + 1];
                    s2 += x[i + 2] * w[k + 2];
                    s3 += x[i + 3] * w[k + 3];
                }
                float sum = (s0 + s1) + (s2 + s3);
                for (; i < inLen; i++, k++) sum += x[i] * w[k];
                result[outOff + j] = SiLU(sum);
            }
        }

        private static void DenseSliceSiLU(float[] x, int xOff, int inLen, float[] w, float[] bias, float[] result)
        {
            int outLen = result.Length, k = 0;
            for (int j = 0; j < outLen; j++)
            {
                float s0 = bias[j], s1 = 0f, s2 = 0f, s3 = 0f;
                int i = 0;
                for (; i + 3 < inLen; i += 4, k += 4)
                {
                    s0 += x[xOff + i] * w[k];
                    s1 += x[xOff + i + 1] * w[k + 1];
                    s2 += x[xOff + i + 2] * w[k + 2];
                    s3 += x[xOff + i + 3] * w[k + 3];
                }
                float sum = (s0 + s1) + (s2 + s3);
                for (; i < inLen; i++, k++) sum += x[xOff + i] * w[k];
                result[j] = SiLU(sum);
            }
        }
    }
}
