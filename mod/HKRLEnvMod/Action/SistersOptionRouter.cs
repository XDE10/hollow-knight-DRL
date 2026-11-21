using System;
using System.IO;

namespace HKRLEnvMod.Action
{
    // A learned choice between two frozen primitive policies every 0.8 game seconds.
    // The clock begins at the first observation, as in the randomized training tapes.
    internal sealed class SistersOptionRouter
    {
        internal const int Magic = 0x354C5149;
        private const int Width = 192, Features = 12;
        private readonly NativeSistersNetwork _alternate;
        private readonly float[] _projection, _mean, _scale, _coef;
        private float _next = -1;
        private bool _useAlternate;

        private SistersOptionRouter(BinaryReader reader, NativeSistersNetwork alternate)
        {
            _alternate = alternate;
            _projection = NativeSistersNetwork.ReadWeights(reader, Width * Features);
            _mean = NativeSistersNetwork.ReadWeights(reader, Features);
            _scale = NativeSistersNetwork.ReadWeights(reader, Features);
            _coef = NativeSistersNetwork.ReadWeights(reader, Features + 1);
            if (Array.Exists(_scale, value => value <= 0))
                throw new InvalidDataException("Invalid option feature scale");
        }

        internal static NativeSistersNetwork Read(BinaryReader reader, out SistersOptionRouter? router)
        {
            int magic = reader.ReadInt32();
            reader.BaseStream.Position = 0;
            router = null;
            if (magic == SistersCompoundOption.Magic)
            {
                reader.ReadInt32();
                var net = SistersCompoundOption.Read(reader, out var option);
                net.Compound = option;
                return net;
            }
            if (magic != Magic) return new NativeSistersNetwork(reader);
            reader.ReadInt32();
            int first = reader.ReadInt32(), second = reader.ReadInt32();
            const int gateBytes = (Width * Features + 3 * Features + 1) * sizeof(float);
            if (first < 20 || second < 20 || (long)first + second + gateBytes + 12 != reader.BaseStream.Length)
                throw new InvalidDataException("Invalid option checkpoint sizes");
            NativeSistersNetwork Actor(int size)
            {
                using var component = new BinaryReader(new MemoryStream(reader.ReadBytes(size), false));
                if (component.ReadInt32() != 0x344C5149)
                    throw new InvalidDataException("Expected a timing-history option policy");
                component.BaseStream.Position = 0;
                return new NativeSistersNetwork(component);
            }
            var primary = Actor(first);
            router = new SistersOptionRouter(reader, Actor(second));
            return primary;
        }

        internal NativeSistersNetwork Select(NativeSistersNetwork active, NativeSistersNetwork trio,
            float[] state, float time)
        {
            if (time >= _next)
            {
                _next = time + .8f;
                _useAlternate = active == trio && Score(trio.TimingEmbedding(state, time)) > 0;
            }
            if (active == trio && _useAlternate) return _alternate;
            _alternate.Remember(state, time);
            return active;
        }

        private double Score(float[] embedding)
        {
            double score = _coef[0];
            for (int j = 0; j < Features; j++)
            {
                double sum = 0;
                for (int k = 0; k < Width; k++) sum += (double)embedding[k] * _projection[k * Features + j];
                score += _coef[j + 1] * Math.Max(-6, Math.Min(6, (sum - _mean[j]) / _scale[j]));
            }
            return score;
        }
    }
}
