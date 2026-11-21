using System;
using System.Globalization;
using System.IO;

namespace HKRLEnvMod.Action
{
    internal sealed class SistersFocusOption
    {
        internal const int Magic = 0x374C5149;
        private const int Features = 16;
        private readonly float[] _contrast;
        private int _startHp = -1;
        private float _extendedAt = -1, _previousTime = -1;
        private int _extensionCount, _extensionInputs, _gains, _debits, _lastHp = -1;
        private float _watchUntil = -1;
        private (int starts, int inputs, int gains, int debits) _written = (-1, -1, -1, -1);

        private SistersFocusOption(float[] contrast) { _contrast = contrast; }

        internal static NativeSistersNetwork Read(BinaryReader reader)
        {
            if (reader.ReadInt32() != Magic) throw new InvalidDataException("Wrong focus policy");
            int n = reader.ReadInt32();
            if (n < 20 || 8L + n + Features * sizeof(float) != reader.BaseStream.Length)
                throw new InvalidDataException("Invalid focus policy length");
            using var leaf = new BinaryReader(new MemoryStream(reader.ReadBytes(n), false));
            if (leaf.ReadInt32() != 0x344C5149) throw new InvalidDataException("Expected a timing history actor");
            leaf.BaseStream.Position = 0;
            var policy = new NativeSistersNetwork(leaf);
            policy.Focus = new SistersFocusOption(NativeSistersNetwork.ReadWeights(reader, Features));
            return policy;
        }

        private double Advantage(float[] s)
        {
            // Same causal basis as focus_features in the offline return estimator.
            var w = _contrast;
            double age = s[178];
            return w[0] + w[1] * age + w[2] * Math.Max(0, age - .6)
                + w[3] * Math.Max(0, age - 1) + w[4] * Math.Max(0, age - 1.3)
                + w[5] * Math.Max(0, age - 1.6) + w[6] * s[4] + w[7] * s[5]
                + w[8] * s[7] + w[9] * s[9] + w[10] * s[12] + w[11] * s[21]
                + w[12] * (s[24] > 0 ? Math.Abs(s[25]) : 2)
                + w[13] * (s[132] > 0 ? Math.Abs(s[133]) : 2)
                + w[14] * s[23] + w[15] * age * s[5];
        }

        internal PrimitiveInput Choose(PrimitiveInput original, float[] state, float time)
        {
            int hp = (int)Math.Round(state[4] * 9);
            bool newAttempt = state[169] < .5f;
            bool continuous = _previousTime >= 0 && time > _previousTime && time - _previousTime <= .081f;
            _previousTime = time;
            if (continuous && time <= _watchUntil)
            {
                _gains += Math.Max(0, hp - _lastHp);
                _debits += Math.Max(0, _lastHp - hp);
            }
            _lastHp = hp;
            if (newAttempt || !continuous)
            {
                _startHp = newAttempt ? hp : -1;
                _extendedAt = -1;
                return original;
            }
            if (_startHp != hp) _startHp = -1;
            if (hp <= 0 || hp >= 9 || _startHp < 0 || state[178] > 1.95f || (original.Buttons & 32) != 0)
                return original;
            if ((_extendedAt >= 0 && time >= _extendedAt + .52f) || Advantage(state) <= 0)
                return original;
            if (_extendedAt < 0)
            {
                _extendedAt = time;
                _watchUntil = time + .52f;
                _extensionCount++;
            }
            _extensionInputs++;
            return new PrimitiveInput(0, 0, 32);
        }

        internal void WriteReceipt(string path, ulong episode)
        {
            var current = (_extensionCount, _extensionInputs, _gains, _debits);
            if (_written == current) return;
            File.WriteAllText(path, string.Format(CultureInfo.InvariantCulture,
                "{0}\n{1}\n{2}\n{3}\n{4}", episode, _extensionCount, _extensionInputs, _gains, _debits));
            _written = current;
        }
    }
}
