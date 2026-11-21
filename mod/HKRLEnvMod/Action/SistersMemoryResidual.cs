using System;
using System.IO;

namespace HKRLEnvMod.Action
{
    // A PyTorch-compatible GRU adjusts all native action scores before their greedy selection.
    internal sealed class SistersMemoryResidual
    {
        internal const int Magic = 0x384C5149;
        private const int Width = 192, Hidden = 24, Gates = 3 * Hidden;
        private readonly float[] _input, _recurrent, _inputBias, _recurrentBias, _head, _headBias;
        private readonly float[] _h = new float[Hidden], _xGates = new float[Gates], _hGates = new float[Gates];
        private readonly float[] _delta;
        private float _start = -1, _previous = -1;

        private SistersMemoryResidual(BinaryReader reader, int actions)
        {
            _input = NativeSistersNetwork.ReadWeights(reader, Gates * Width);
            _recurrent = NativeSistersNetwork.ReadWeights(reader, Gates * Hidden);
            _inputBias = NativeSistersNetwork.ReadWeights(reader, Gates);
            _recurrentBias = NativeSistersNetwork.ReadWeights(reader, Gates);
            _head = NativeSistersNetwork.ReadWeights(reader, actions * Hidden);
            _headBias = NativeSistersNetwork.ReadWeights(reader, actions);
            _delta = new float[actions];
        }

        internal static NativeSistersNetwork Read(BinaryReader reader)
        {
            if (reader.ReadInt32() != Magic) throw new InvalidDataException("Wrong recurrent control policy");
            int n = reader.ReadInt32();
            if (n < 20 || n > reader.BaseStream.Length - 8) throw new InvalidDataException("Invalid base controller size");
            using var leaf = new BinaryReader(new MemoryStream(reader.ReadBytes(n), false));
            if (leaf.ReadInt32() != 0x344C5149) throw new InvalidDataException("Expected a timing history controller");
            leaf.BaseStream.Position = 0;
            var actor = new NativeSistersNetwork(leaf);
            long count = Gates * (Width + Hidden + 2) + (Hidden + 1) * actor.ActionCount;
            if (reader.BaseStream.Length - reader.BaseStream.Position != count * sizeof(float))
                throw new InvalidDataException("Invalid recurrent control weights");
            actor.Memory = new SistersMemoryResidual(reader, actor.ActionCount);
            return actor;
        }

        private static float Sigmoid(float x) => (float)(1.0 / (1.0 + Math.Exp(-x)));

        internal void Adjust(float[] state, float time, float[] scores)
        {
            if (_previous < 0 || time <= _previous || time - _previous > .081f || time >= _start + 3.2f)
            {
                Array.Clear(_h, 0, Hidden);
                _start = time;
            }
            _previous = time;
            NativeSistersNetwork.Dense(state, _input, _inputBias, _xGates, false);
            NativeSistersNetwork.Dense(_h, _recurrent, _recurrentBias, _hGates, false);
            for (int i = 0; i < Hidden; i++)
            {
                float reset = Sigmoid(_xGates[i] + _hGates[i]);
                float update = Sigmoid(_xGates[Hidden+i] + _hGates[Hidden+i]);
                float next = (float)Math.Tanh(_xGates[2*Hidden+i] + reset * _hGates[2*Hidden+i]);
                _h[i] = update * _h[i] + (1 - update) * next;
            }
            NativeSistersNetwork.Dense(_h, _head, _headBias, _delta, false);
            for (int i = 0; i < scores.Length; i++) scores[i] += 3f * (float)Math.Tanh(_delta[i]);
        }
    }
}
