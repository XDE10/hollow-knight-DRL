using System;
using System.IO;

namespace HKRLEnvMod.Action
{
    // PyTorch GRU gate order is reset, update, new. The encoded actor folds its
    // feature projection into the score weights, so it needs only one GRU pass.
    internal sealed class SistersMemoryResidual
    {
        internal const int Magic = 0x384C5149;
        internal const int EncodedMagic = 0x3C4C5149;
        private const int Width = 192, LegacyHidden = 24;
        private readonly int _hidden;
        private readonly bool _encoded;
        private readonly float[] _input, _recurrent, _inputBias, _recurrentBias, _head, _headBias;
        private readonly float[] _h, _xGates, _hGates;
        private readonly float[] _delta;
        private float _start = -1, _previous = -1;
        private int _decisions, _changed;

        internal void WriteReceipt(string path, ulong episode)
        {
            File.WriteAllText(path, string.Join("\n", episode, _decisions, _changed));
        }

        private SistersMemoryResidual(BinaryReader reader, int actions, int hidden, bool encoded)
        {
            _hidden = hidden;
            _encoded = encoded;
            int gates = 3 * hidden;
            _h = new float[hidden];
            _xGates = new float[gates];
            _hGates = new float[gates];
            _input = NativeSistersNetwork.ReadWeights(reader, gates * Width);
            _recurrent = NativeSistersNetwork.ReadWeights(reader, gates * hidden);
            _inputBias = NativeSistersNetwork.ReadWeights(reader, gates);
            _recurrentBias = NativeSistersNetwork.ReadWeights(reader, gates);
            _head = NativeSistersNetwork.ReadWeights(reader, actions * hidden);
            _headBias = NativeSistersNetwork.ReadWeights(reader, actions);
            _delta = new float[actions];
        }

        internal static NativeSistersNetwork Read(BinaryReader reader)
        {
            int magic = reader.ReadInt32();
            bool encoded = magic == EncodedMagic;
            if (!encoded && magic != Magic) throw new InvalidDataException("Wrong recurrent control policy");
            int n = reader.ReadInt32();
            int hidden = encoded ? reader.ReadInt32() : LegacyHidden;
            if (hidden < 1 || hidden > 256) throw new InvalidDataException("Invalid recurrent hidden size");
            if (n < 20 || n > reader.BaseStream.Length - reader.BaseStream.Position)
                throw new InvalidDataException("Invalid base controller size");
            using var leaf = new BinaryReader(new MemoryStream(reader.ReadBytes(n), false));
            if (leaf.ReadInt32() != 0x344C5149) throw new InvalidDataException("Expected a timing history controller");
            leaf.BaseStream.Position = 0;
            var actor = new NativeSistersNetwork(leaf);
            long count = 3L * hidden * (Width + hidden + 2) + (hidden + 1L) * actor.ActionCount;
            if (reader.BaseStream.Length - reader.BaseStream.Position != count * sizeof(float))
                throw new InvalidDataException("Invalid recurrent control weights");
            actor.Memory = new SistersMemoryResidual(reader, actor.ActionCount, hidden, encoded);
            return actor;
        }

        private static float Sigmoid(float x) => (float)(1.0 / (1.0 + Math.Exp(-x)));

        internal void Adjust(float[] state, float[] embedding, float time, float[] scores)
        {
            if (_previous < 0 || time <= _previous || time - _previous > .081f || time >= _start + 3.2f)
            {
                Array.Clear(_h, 0, _hidden);
                _start = time;
            }
            _previous = time;
            NativeSistersNetwork.Dense(_encoded ? embedding : state, _input, _inputBias, _xGates, false);
            NativeSistersNetwork.Dense(_h, _recurrent, _recurrentBias, _hGates, false);
            for (int i = 0; i < _hidden; i++)
            {
                float reset = Sigmoid(_xGates[i] + _hGates[i]);
                float update = Sigmoid(_xGates[_hidden + i] + _hGates[_hidden + i]);
                float next = (float)Math.Tanh(_xGates[2 * _hidden + i] + reset * _hGates[2 * _hidden + i]);
                _h[i] = update * _h[i] + (1 - update) * next;
            }
            NativeSistersNetwork.Dense(_h, _head, _headBias, _delta, false);
            int original = 0, adjusted = 0;
            float originalMax = scores[0];
            for (int i = 0; i < scores.Length; i++)
            {
                if (scores[i] > originalMax) { original = i; originalMax = scores[i]; }
                scores[i] += _encoded ? _delta[i] : 3f * (float)Math.Tanh(_delta[i]);
                if (scores[i] > scores[adjusted]) adjusted = i;
            }
            _decisions++;
            if (original != adjusted) _changed++;
        }
    }
}
