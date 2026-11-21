using System;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;

namespace HKRLEnvMod.Action
{
    // Keep one feasible native key for each horizontal direction and maneuver type.
    // All six scores and the chance of each sampled action are written in development.
    internal sealed class SistersDirectionalOptions
    {
        internal const int Magic = 0x3A4C5149, HeadMagic = 0x33504753, TapeMagic = 0x36434B48;
        private const int Groups = 6, Width = 193;
        internal readonly byte[] Digest;
        internal readonly float Interval;
        private readonly float[] _risk, _weights;
        private readonly int[] _slots = new int[Groups];
        private readonly double[] _prob = new double[Groups];
        private float _decisionAt, _release;
        private PrimitiveInput _onset;
        private int _count, _alternate, _changed;
        private (int, int, int)? _written;

        internal SistersDirectionalOptions(byte[] head)
        {
            using var input = new BinaryReader(new MemoryStream(head, false));
            if (head.Length != 8 + (Groups + 1) * Width * sizeof(float) || input.ReadInt32() != HeadMagic)
                throw new InvalidDataException("Invalid directional option policy");
            Interval = input.ReadSingle();
            if (!(Interval >= .32f && Interval <= 8.4f)) throw new InvalidDataException("Invalid native decision interval");
            _risk = NativeSistersNetwork.ReadWeights(input, Width);
            _weights = NativeSistersNetwork.ReadWeights(input, Groups * Width);
            using var sha = SHA256.Create();
            Digest = sha.ComputeHash(head);
        }

        internal static NativeSistersNetwork Read(BinaryReader reader)
        {
            if (reader.ReadInt32() != Magic) throw new InvalidDataException("Missing directional actor");
            int size = reader.ReadInt32(), headSize = 8 + (Groups + 1) * Width * sizeof(float);
            if (size < 20 || reader.BaseStream.Length != 8L + size + headSize) throw new InvalidDataException("Invalid directional actor length");
            using var inner = new BinaryReader(new MemoryStream(reader.ReadBytes(size), false));
            if (inner.ReadInt32() != 0x344C5149) throw new InvalidDataException("Expected a native history actor");
            inner.BaseStream.Position = 0;
            var actor = new NativeSistersNetwork(inner);
            actor.Directional = new SistersDirectionalOptions(reader.ReadBytes(headSize));
            return actor;
        }

        private static double Linear(float[] w, int offset, float[] hidden)
        {
            double value = w[offset];
            for (int i = 0; i < hidden.Length; i++) value += (double)w[offset + i + 1] * hidden[i];
            return value;
        }

        internal void Reset()
        {
            _decisionAt = _release = 0;
            _count = _alternate = _changed = 0;
            _written = null;
        }

        internal void WriteReceipt(string path, ulong episode)
        {
            var counts = (_count, _alternate, _changed);
            if (_written == counts) return;
            File.WriteAllText(path, string.Format(CultureInfo.InvariantCulture,
                "{0}\n{1}\n{2}\n{3}", episode, _count, _alternate, _changed));
            _written = counts;
        }

        internal PrimitiveInput Choose(NativeSistersNetwork policy, float[] state, float time,
                                      Random? random = null, BinaryWriter? tape = null)
        {
            var original = policy.Command(policy.BestIndex);
            if (time < _release)
            {
                var input = policy.ContinueOption(_onset, original);
                if (input.MovementX != original.MovementX || input.AimY != original.AimY || input.Buttons != original.Buttons) _changed++;
                return input;
            }
            if (time < _decisionAt || state[23] < .5f || state[9] > .5f) return original;
            double risk = Linear(_risk, 0, policy.Embedding);
            if (risk < 0) return original;
            int primary = (original.MovementX + 1) * 2, family = SistersCompoundOption.Maneuver(original);
            for (int g = 0; g < Groups; g++) _slots[g] = -1;
            _slots[primary] = policy.BestIndex;
            for (int i = 0; i < policy.ActionCount; i++)
            {
                var key = policy.Command(i);
                int g = (key.MovementX + 1) * 2 + (SistersCompoundOption.Maneuver(key) == family ? 0 : 1);
                if (g == primary || !SistersCompoundOption.Available(key, state)) continue;
                if (_slots[g] < 0 || policy.Score(i) > policy.Score(_slots[g])) _slots[g] = i;
            }
            int valid = 0;
            for (int g = 0; g < Groups; g++) if (_slots[g] >= 0) valid++;
            if (valid < 2) return original;
            int selected = primary;
            double maximum = double.NegativeInfinity;
            for (int g = 0; g < Groups; g++)
            {
                _prob[g] = _slots[g] < 0 ? double.NegativeInfinity : Linear(_weights, g * Width, policy.Embedding)
                    + (g == primary ? Math.Log(3.0 * (valid - 1)) : 0);
                if (_prob[g] > maximum) { maximum = _prob[g]; selected = g; }
            }
            double sum = 0;
            for (int g = 0; g < Groups; g++)
            {
                _prob[g] = Math.Exp(_prob[g] - maximum);
                sum += _prob[g];
            }
            for (int g = 0; g < Groups; g++) _prob[g] = _slots[g] < 0 ? 0 : .95 * _prob[g] / sum + .05 / valid;
            if (random != null)
            {
                double draw = random.NextDouble(), acc = 0;
                for (int g = 0; g < Groups; g++)
                {
                    if (_slots[g] < 0) continue;
                    selected = g; acc += _prob[g];
                    if (draw < acc) break;
                }
            }
            if (tape != null)
            {
                if (random == null) throw new InvalidOperationException("Deterministic input cannot produce a stochastic rollout");
                tape.Write(time); tape.Write(primary); tape.Write(selected); tape.Write((float)risk);
                for (int g = 0; g < Groups; g++) tape.Write(_slots[g]);
                for (int g = 0; g < Groups; g++) tape.Write((float)_prob[g]);
            }
            _count++; _decisionAt = time + Interval;
            if (selected == primary) return original;
            _alternate++; _changed++; _release = time + .12f;
            return _onset = policy.Command(_slots[selected]);
        }
    }
}
