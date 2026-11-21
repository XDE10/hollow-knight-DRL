using System;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using HKRLEnvMod.Observation;

namespace HKRLEnvMod.Action
{
    internal interface ISistersLocalActor : IDisposable
    {
        PrimitiveInput Input { get; }
        bool NeedsObservation(ulong tick, ulong episode, ulong commits);
        void Decide(ObservationSnapshot obs, ulong tick);
        void Record(float time);
        void Flush();
    }

    // Direct learned primitive control. The heartbeat only bounds how long a disconnected
    // actor may run. The tactical controller is mutually exclusive with this actor.
    internal sealed class SistersNativeActor : ISistersLocalActor
    {
        private readonly string _path;
        private readonly string _ledger;
        private readonly SistersNativeDemonstrator? _demonstrator;
        private readonly float[] _raw = new float[56];
        private readonly float[] _boss = new float[3];
        private readonly float[] _disc = new float[2];
        private NativeSistersNetwork? _net;
        private NativeSistersNetwork? _trioNet;
        private SistersOptionRouter? _router;
        private SistersTimingState? _timing;
        private ulong _episode, _tick, _commits;
        private int _count, _attack, _cast, _dash, _art;
        private string _digest = "";
        public PrimitiveInput Input { get; private set; }

        private SistersNativeActor(string path, string ledger, SistersNativeDemonstrator? demonstrator)
        {
            _path = path;
            _ledger = ledger;
            _demonstrator = demonstrator;
        }

        public static ISistersLocalActor? FromEnvironment()
        {
            var policy = Environment.GetEnvironmentVariable("HKRL_NATIVE_ACTOR");
            if (string.IsNullOrWhiteSpace(policy)) return SistersTactics.FromEnvironment();
            var demonstrator = SistersNativeDemonstrator.FromEnvironment();
            if (demonstrator == null && !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("HKRL_TACTICS_POLICY")))
                throw new InvalidOperationException("Select only one local Sisters actor");
            var ledger = Environment.GetEnvironmentVariable("HKRL_NATIVE_ACTOR_LEDGER");
            if (string.IsNullOrWhiteSpace(ledger)) throw new InvalidOperationException("Missing native actor ledger path");
            if (Environment.GetEnvironmentVariable("HKRL_COMPACT_RENDER") == "1")
            {
                UnityEngine.QualitySettings.vSyncCount = 0;
                UnityEngine.Application.targetFrameRate = 120;
            }
            return new SistersNativeActor(policy!, ledger!, demonstrator);
        }

        public bool NeedsObservation(ulong tick, ulong episode, ulong commits)
        {
            if (_net != null && episode == _episode && (tick-_tick < 2 || commits == _commits)) return false;
            _commits = commits;
            return true;
        }

        public void Decide(ObservationSnapshot obs, ulong tick)
        {
            if (_net == null || _episode != obs.Global.EpisodeId)
            {
                var bytes = File.ReadAllBytes(_path);
                using (var sha = SHA256.Create())
                    _digest = BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
                _net = NativeSistersNetwork.Load(bytes, out _trioNet, out _router);
                _episode = obs.Global.EpisodeId;
                _count = _attack = _cast = _dash = _art = 0;
                _demonstrator?.Reset(_episode);
                _demonstrator?.NeuralTrial?.Reset(_episode, bytes);
                _timing = _net.UsesTiming || _demonstrator?.RecordsTiming == true ? new SistersTimingState() : null;
            }
            _tick = tick;
            if (!_net.UsesTiming || _demonstrator?.RecordsRawFeatures == true)
                SistersPolicyHead.WriteFeatures(obs, _raw, _boss, _disc);
            _timing?.Observe(obs);
            var state = _net.UsesTiming ? _timing!.Values : _raw;
            var active = _trioNet != null && state[SistersTimingState.TrioFlag] > .5f ? _trioNet : _net;
            if (_router != null) active = _router.Select(active, _trioNet!, state, obs.Global.TimeInEpisode);
            if (active != _trioNet) _trioNet?.Remember(state, obs.Global.TimeInEpisode);
            if (_demonstrator?.NeuralTrial != null)
                active = _demonstrator.NeuralTrial.Select(active, active == _trioNet, state, obs.Global.TimeInEpisode);
            Input = active.Choose(state, obs.Global.TimeInEpisode);
            if (_demonstrator != null) Input = _demonstrator.Label(obs, tick, _raw, Input, _timing?.Values, active);
            _timing?.Commit(Input, obs.Global.TimeInEpisode);
            if (!_net.UsesTiming) _net.Commit(Input, obs.Global.TimeInEpisode);
            _count++;
            if ((Input.Buttons & 8) != 0) _attack++;
            if ((Input.Buttons & 16) != 0) _cast++;
            if ((Input.Buttons & 4) != 0) _dash++;
            if ((Input.Buttons & 384) != 0) _art++;
        }

        public void Record(float time) { }

        public void Flush()
        {
            _demonstrator?.Flush();
            if (_count == 0) return;
            _trioNet?.Focus?.WriteReceipt(_ledger + ".focus", _episode);
            File.WriteAllText(_ledger, string.Join("\n", _episode.ToString(CultureInfo.InvariantCulture),
                _digest, _count, _attack, _cast, _dash, _art));
        }

        public void Dispose() { _demonstrator?.Dispose(); }
    }

    internal sealed class NativeSistersNetwork
    {
        private const int Physical = 56;
        internal bool UsesTiming { get; }
        private static readonly uint[] Pulses = { 1, 4, 8, 16, 32, 256 };
        private static readonly uint[] Holds = { 2, 128, 32 };
        private readonly PrimitiveInput[] _actions;
        private readonly float[][] _w = new float[6][];
        private readonly float[] _history = new float[Physical * 4];
        private readonly float[] _x, _h, _z;
        private readonly float[][] _entityWeights = new float[4][];
        private readonly SistersMotionHistory? _motionHistory;
        private readonly float[] _scores, _pulseAt = new float[6], _holdAt = new float[3];
        private uint _previous;
        private int _index;
        private int _best;
        internal SistersCompoundOption? Compound { get; set; }
        internal SistersFocusOption? Focus { get; set; }
        internal int BestIndex => _best;
        internal float Score(int index) => _scores[index];
        internal float[] Embedding => _z;

        internal static NativeSistersNetwork Load(byte[] bytes, out NativeSistersNetwork? trio)
        {
            var opener = Load(bytes, out trio, out var router);
            if (router != null) throw new InvalidDataException("A routed policy needs its option selector");
            return opener;
        }

        internal static NativeSistersNetwork Load(byte[] bytes, out NativeSistersNetwork? trio,
            out SistersOptionRouter? router)
        {
            router = null;
            trio = null;
            using var source = new MemoryStream(bytes, false);
            using var reader = new BinaryReader(source);
            int magic = reader.ReadInt32();
            if (magic != 0x334C5149)
            {
                source.Position = 0;
                return new NativeSistersNetwork(reader);
            }
            int firstSize = reader.ReadInt32(), trioSize = reader.ReadInt32();
            if (firstSize < 20 || trioSize < 20 || (long)firstSize + trioSize != source.Length - source.Position)
                throw new InvalidDataException("Invalid Sisters phase checkpoint lengths");
            using var firstReader = new BinaryReader(new MemoryStream(reader.ReadBytes(firstSize), false));
            using var trioReader = new BinaryReader(new MemoryStream(reader.ReadBytes(trioSize), false));
            var opener = new NativeSistersNetwork(firstReader);
            trio = SistersOptionRouter.Read(trioReader, out router);
            if (!opener.UsesTiming || !trio.UsesTiming)
                throw new InvalidDataException("Both phase policies require the timing observations");
            return opener;
        }

        internal NativeSistersNetwork(BinaryReader reader)
        {
            int magic = reader.ReadInt32();
            bool usesHistory = magic == 0x344C5149;
            UsesTiming = magic == 0x324C5149 || usesHistory;
            int obsWidth = UsesTiming ? SistersTimingState.Width : 180;
            int hidden1 = UsesTiming ? 256 : 160, hidden2 = UsesTiming ? 192 : 128;
            if ((!UsesTiming && magic != 0x314C5149) || reader.ReadInt32() != obsWidth ||
                reader.ReadInt32() != hidden1 || reader.ReadInt32() != hidden2)
                throw new InvalidDataException("Invalid native actor architecture");
            _x = new float[obsWidth + (UsesTiming ? 72 : 0) + (usesHistory ? 64 : 0)];
            _h = new float[hidden1];
            _z = new float[hidden2];
            int n = reader.ReadInt32();
            if (n < 1 || n > 4096) throw new InvalidDataException("Invalid action count");
            _scores = new float[n];
            _actions = new PrimitiveInput[n];
            for (int i=0; i<n; i++)
            {
                int move=reader.ReadInt32(), aim=reader.ReadInt32(), buttons=reader.ReadInt32();
                if (Math.Abs(move)>1 || Math.Abs(aim)>1 || buttons<0 || buttons>511)
                    throw new InvalidDataException("Invalid primitive in native actor catalogue");
                _actions[i]=new PrimitiveInput(move,aim,(uint)buttons);
            }
            if (UsesTiming)
            {
                int[] shape = {36*48, 48, 14*24, 24};
                for (int i = 0; i < shape.Length; i++) _entityWeights[i] = ReadWeights(reader, shape[i]);
            }
            if (usesHistory)
                _motionHistory = new SistersMotionHistory(ReadWeights(reader, (obsWidth + 1) * 32), ReadWeights(reader, 32));
            int[] sizes = {_x.Length*hidden1, hidden1, hidden1*hidden2, hidden2, hidden2*n, n};
            for (int layer = 0; layer < sizes.Length; layer++) _w[layer] = ReadWeights(reader, sizes[layer]);
            if(reader.BaseStream.Position!=reader.BaseStream.Length) throw new InvalidDataException("Trailing actor bytes");
            for (int i=0;i<_pulseAt.Length;i++) _pulseAt[i]=-100;
            for (int i=0;i<_holdAt.Length;i++) _holdAt[i]=-100;
        }

        internal PrimitiveInput Choose(float[] raw, float time)
        {
            if (UsesTiming)
            {
                TimingEmbedding(raw, time);
                Remember(raw, time);
            }
            else
            {
                PrepareHistory(raw, time);
                Dense(_x,_w[0],_w[1],_h,true);
                Dense(_h,_w[2],_w[3],_z,true);
            }
            Dense(_z,_w[4],_w[5],_scores,false);
            int best=0;
            for(int i=1;i<_scores.Length;i++) if (_scores[i]>_scores[best]) best=i;
            _best = best;
            _index++;
            if (Focus != null) return Focus.Choose(_actions[best], raw, time);
            return Compound == null ? _actions[best] : Compound.Choose(this, raw, time);
        }

        internal PrimitiveInput Command(int index) => _actions[index];

        internal int CompoundAlternative(float[] state)
        {
            int best = -1;
            int original = SistersCompoundOption.Maneuver(_actions[_best]);
            for (int i = 0; i < _actions.Length; i++)
                if (SistersCompoundOption.Maneuver(_actions[i]) != original &&
                    SistersCompoundOption.Available(_actions[i], state) && (best < 0 || _scores[i] > _scores[best])) best = i;
            return best;
        }

        internal PrimitiveInput ContinueOption(PrimitiveInput onset, PrimitiveInput greedy)
        {
            int best = -1;
            // Hold lateral intent and jump hold. Fresh jump/dash/cast pulses after
            // the onset are available only when the underlying network requests them.
            const uint fresh = 1u | 4u | 16u;
            for (int i = 0; i < _actions.Length; i++)
            {
                var candidate = _actions[i];
                if (candidate.MovementX != onset.MovementX || ((candidate.Buttons ^ onset.Buttons) & 2u) != 0 ||
                    (candidate.Buttons & fresh & ~greedy.Buttons) != 0) continue;
                if (best < 0 || _scores[i] > _scores[best]) best = i;
            }
            return best >= 0 ? _actions[best] : greedy;
        }

        internal PrimitiveInput BestHorizontal(int movement)
        {
            int best = -1;
            for (int i = 0; i < _actions.Length; i++)
                if (_actions[i].MovementX == movement && (best < 0 || _scores[i] > _scores[best]))
                    best = i;
            if (best < 0) throw new InvalidOperationException("Unavailable recorded movement");
            return _actions[best];
        }

        internal bool MovementPair(out int first, out int other, out float gap, bool horizontalOnly = false)
        {
            first = _best;
            other = -1;
            var greedy = _actions[first];
            for (int i = 0; i < _actions.Length; i++)
            {
                var candidate = _actions[i];
                if (horizontalOnly && candidate.MovementX == greedy.MovementX) continue;
                if (candidate.MovementX == greedy.MovementX &&
                    ((candidate.Buttons & 3) != 0) == ((greedy.Buttons & 3) != 0) &&
                    (candidate.Buttons & 4) == (greedy.Buttons & 4)) continue;
                if (other < 0 || _scores[i] > _scores[other]) other = i;
            }
            gap = other < 0 ? float.PositiveInfinity : _scores[first] - _scores[other];
            return other >= 0;
        }

        internal float[] TimingEmbedding(float[] raw, float time)
        {
            Array.Copy(raw, _x, SistersTimingState.Width);
            Pool(raw, 24, 3, 36, _entityWeights[0], _entityWeights[1], 192);
            Pool(raw, 132, 2, 14, _entityWeights[2], _entityWeights[3], 240);
            _motionHistory?.Encode(time, _x, 264);
            Dense(_x,_w[0],_w[1],_h,true);
            Dense(_h,_w[2],_w[3],_z,true);
            return _z;
        }

        internal void Remember(float[] raw, float time) => _motionHistory?.Remember(raw, time);

        internal static float[] ReadWeights(BinaryReader reader, int count)
        {
            var w = new float[count];
            for (int i=0; i<count; i++)
            {
                w[i] = reader.ReadSingle();
                if (float.IsNaN(w[i]) || float.IsInfinity(w[i])) throw new InvalidDataException("Non-finite actor weight");
            }
            return w;
        }

        private void Pool(float[] raw, int start, int slots, int width, float[] weights, float[] bias, int output)
        {
            Array.Clear(_x, output, bias.Length);
            for (int slot = 0; slot < slots; slot++)
            {
                int k = start + slot*width;
                if (raw[k] == 0)
                {
                    Array.Clear(_x, k, width);
                    continue;
                }
                for (int j=0; j<bias.Length; j++)
                {
                    float sum=bias[j];
                    for (int i=0; i<width; i++) sum += raw[k+i]*weights[j*width+i];
                    _x[output+j]=Math.Max(_x[output+j], sum);
                }
            }
        }

        private void PrepareHistory(float[] raw, float time)
        {
            if (_index==0)
                for (int i=0; i<4; i++) Array.Copy(raw,0,_history,i*Physical,Physical);
            Array.Copy(raw,_x,Physical);
            Array.Copy(_history,((_index+3)%4)*Physical,_x,Physical,Physical);
            Array.Copy(_history,(_index%4)*Physical,_x,2*Physical,Physical);
            Array.Copy(raw,0,_history,(_index%4)*Physical,Physical);
            int k=3*Physical;
            for (int j=0;j<Pulses.Length;j++) _x[k++]=Math.Min(time-_pulseAt[j],1f);
            for (int j=0;j<Holds.Length;j++)
            {
                bool held=(_previous & Holds[j])!=0;
                _x[k++]=held?1f:0f;
                _x[k++]=held?Math.Min(time-_holdAt[j],1f):0f;
            }
        }

        internal void Commit(PrimitiveInput action, float time)
        {
            uint press=action.Buttons & ~_previous;
            for(int j=0;j<Pulses.Length;j++) if ((press & Pulses[j])!=0) _pulseAt[j]=time;
            for(int j=0;j<Holds.Length;j++) if ((press & Holds[j])!=0) _holdAt[j]=time;
            _previous=action.Buttons;
        }

        private static void Dense(float[] x,float[] w,float[] bias,float[] result,bool relu)
        {
            int k=0;
            for(int j=0;j<result.Length;j++)
            {
                float sum=bias[j];
                for(int i=0;i<x.Length;i++) sum+=w[k++]*x[i];
                result[j]=relu?Math.Max(0f,sum):sum;
            }
        }
    }
}
