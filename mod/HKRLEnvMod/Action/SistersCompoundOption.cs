using System;
using System.IO;
using System.Globalization;
using System.Security.Cryptography;

namespace HKRLEnvMod.Action
{
    // Both policies use native joint scores. A fitted risk readout schedules a decision.
    // A second fitted readout (or a logged random draw in development) selects the option.
    internal sealed class SistersCompoundOption
    {
        internal const int Magic = 0x364C5149;
        internal const int PolicyGradientMagic = 0x394C5149;
        internal const int SamplingMagic = 0x31504753;
        private const int RiskMagic = 0x314B5352;
        private readonly float[] _risk;
        private readonly float[]? _switch;
        private readonly float _interval;
        internal bool LearnedSampling { get; }
        internal float Interval => _interval;
        internal byte[]? SamplingDigest { get; private set; }
        private float _next, _release;
        private PrimitiveInput _start;
        private int _decisions, _alternates, _changed;
        private (int, int, int)? _recorded;

        private SistersCompoundOption(float[] risk, float[]? selector, float interval = 8.4f, bool learnedSampling = false)
        {
            _risk = risk;
            _switch = selector;
            _interval = interval;
            LearnedSampling = learnedSampling;
        }

        internal static SistersCompoundOption SamplePolicy(string path)
        {
            var bytes = File.ReadAllBytes(path);
            using var reader = new BinaryReader(new MemoryStream(bytes, false));
            if (reader.BaseStream.Length != 8 + 386 * sizeof(float))
                throw new InvalidDataException("Invalid sampled option policy length");
            var option = ReadSamplingHead(reader);
            using var sha = SHA256.Create();
            option.SamplingDigest = sha.ComputeHash(bytes);
            return option;
        }

        private static SistersCompoundOption ReadSamplingHead(BinaryReader reader)
        {
            if (reader.ReadInt32() != SamplingMagic) throw new InvalidDataException("Missing sampled option head");
            float interval = reader.ReadSingle();
            if (!(interval >= .32f && interval <= 8.4f)) throw new InvalidDataException("Invalid option interval");
            return new SistersCompoundOption(NativeSistersNetwork.ReadWeights(reader, 193),
                NativeSistersNetwork.ReadWeights(reader, 193), interval, true);
        }

        internal static SistersCompoundOption Training(string path)
        {
            using var reader = new BinaryReader(File.OpenRead(path));
            if (reader.ReadInt32() != RiskMagic || reader.BaseStream.Length != 4 + 193 * sizeof(float))
                throw new InvalidDataException("Invalid neural risk readout");
            return new SistersCompoundOption(NativeSistersNetwork.ReadWeights(reader, 193), null);
        }

        internal static NativeSistersNetwork Read(BinaryReader reader, out SistersCompoundOption option, bool policyGradient = false)
        {
            int length = reader.ReadInt32();
            if (length < 20 || (long)length + 8 + 386 * sizeof(float) + (policyGradient ? 8 : 0) != reader.BaseStream.Length)
                throw new InvalidDataException("Invalid compound policy lengths");
            using var input = new BinaryReader(new MemoryStream(reader.ReadBytes(length), false));
            if (input.ReadInt32() != 0x344C5149) throw new InvalidDataException("Expected a history policy");
            input.BaseStream.Position = 0;
            var primary = new NativeSistersNetwork(input);
            option = policyGradient ? ReadSamplingHead(reader) : new SistersCompoundOption(NativeSistersNetwork.ReadWeights(reader, 193),
                NativeSistersNetwork.ReadWeights(reader, 193));
            return primary;
        }

        private static double Predict(float[] weights, float[] hidden)
        {
            double value = weights[0];
            for (int i = 0; i < hidden.Length; i++) value += (double)weights[i + 1] * hidden[i];
            return value;
        }

        internal void Reset()
        {
            _next = _release = 0;
            _decisions = _alternates = _changed = 0;
            _recorded = null;
        }

        internal void WriteReceipt(string path, ulong episode)
        {
            var counts = (_decisions, _alternates, _changed);
            if (!LearnedSampling || counts == _recorded) return;
            File.WriteAllText(path, string.Format(CultureInfo.InvariantCulture,
                "{0}\n{1}\n{2}\n{3}", episode, _decisions, _alternates, _changed));
            _recorded = counts;
        }

        internal PrimitiveInput Choose(NativeSistersNetwork policy, float[] state, float time,
            Random? random = null, BinaryWriter? writer = null)
        {
            var original = policy.Command(policy.BestIndex);
            if (time < _release)
            {
                var issued = policy.ContinueOption(_start, original);
                if (issued.MovementX != original.MovementX || issued.AimY != original.AimY || issued.Buttons != original.Buttons) _changed++;
                return issued;
            }
            if (time < _next || state[9] > .5f || state[23] < .5f) return original;
            double risk = Predict(_risk, policy.Embedding);
            if (risk < 0) return original;
            int other = policy.CompoundAlternative(state);
            if (other < 0) return original;
            double score = _switch == null ? 0 : Predict(_switch, policy.Embedding);
            double probability = LearnedSampling ? Math.Max(.05, Math.Min(.95, 1.0 / (1.0 + Math.Exp(-score)))) : .5;
            bool alternate = random == null ? score > 0 :
                LearnedSampling ? random.NextDouble() < probability : random.Next(2) == 1;
            _next = time + _interval;
            _decisions++;
            if (writer != null)
            {
                writer.Write(time);
                writer.Write(policy.BestIndex);
                writer.Write(other);
                writer.Write(alternate ? 1 : 0);
                writer.Write((float)probability);
                writer.Write(policy.Score(policy.BestIndex) - policy.Score(other));
                writer.Write(policy.Score(policy.BestIndex));
                writer.Write(policy.Score(other));
                writer.Write((float)risk);
            }
            if (!alternate) return original;
            _alternates++;
            _changed++;
            _release = time + .32f;
            _start = policy.Command(other);
            return _start;
        }

        internal static int Maneuver(PrimitiveInput p)
        {
            int jump = (p.Buttons & 3) != 0 ? 1 : 0;
            int dash = (p.Buttons & 4) != 0 ? 2 : 0;
            int dive = (p.Buttons & 16) != 0 && p.AimY < 0 ? 4 : 0;
            return jump | dash | dive;
        }

        internal static bool Available(PrimitiveInput p, float[] state)
        {
            if ((p.Buttons & 1) != 0 && (state[164] > .5f || (state[7] < .5f && state[8] < .5f))) return false;
            if ((p.Buttons & 3) == 2 && state[165] < .5f) return false;
            if ((p.Buttons & 4) != 0 && (state[13] < .5f || state[166] > .5f)) return false;
            if ((p.Buttons & 16) != 0 && (state[11] < .5f || state[5] < 33f/99f || state[168] > .5f)) return false;
            return true;
        }
    }
}
