using System;
using System.Globalization;
using System.IO;

namespace HKRLEnvMod.Action
{
    // Four native observation boundaries and a PyTorch-compatible GRU. The cloud
    // dynamics ensemble is only needed for training the student's preferences.
    internal sealed class SistersWorldStudent
    {
        private const int Width = SistersPolicyHead.Features;
        private const int Hidden = 32;
        private const int Gates = 3 * Hidden;
        private const int Options = SistersPolicyHead.Actions;
        private const int History = 4;
        private const int Recurrent = Gates * Width;
        private const int InputBias = Recurrent + Gates * Hidden;
        private const int HiddenBias = InputBias + Gates;
        private const int Output = HiddenBias + Gates;
        private readonly float[] _weights;
        private readonly float[] _history = new float[History * Width];
        private readonly float[] _hidden = new float[Hidden];
        private readonly float[] _inputGates = new float[Gates];
        private readonly float[] _hiddenGates = new float[Gates];
        private readonly float[] _values = new float[Options];
        private bool _initialized;

        public SistersWorldStudent(string path)
        {
            var lines = File.ReadAllLines(path);
            if (lines.Length != Output + Options * (Hidden + 1))
                throw new InvalidDataException("Unexpected world student weight count.");
            _weights = Array.ConvertAll(lines, text => float.Parse(text, CultureInfo.InvariantCulture));
        }

        private static float Sigmoid(float value) =>
            (float)(1.0 / (1.0 + Math.Exp(-Math.Max(-30f, Math.Min(30f, value)))));

        public float Value(int option) => _values[option];

        public void Observe(float[] features)
        {
            if (!_initialized)
            {
                for (int frame = 0; frame < History; frame++)
                    Array.Copy(features, 0, _history, frame * Width, Width);
                _initialized = true;
            }
            else
            {
                Array.Copy(_history, Width, _history, 0, (History - 1) * Width);
                Array.Copy(features, 0, _history, (History - 1) * Width, Width);
            }
            Array.Clear(_hidden, 0, Hidden);
            for (int frame = 0; frame < History; frame++)
            {
                int inputOffset = frame * Width;
                for (int gate = 0; gate < Gates; gate++)
                {
                    float input = _weights[InputBias + gate], recurrent = _weights[HiddenBias + gate];
                    for (int col = 0; col < Width; col++)
                        input += _weights[gate * Width + col] * _history[inputOffset + col];
                    for (int col = 0; col < Hidden; col++)
                        recurrent += _weights[Recurrent + gate * Hidden + col] * _hidden[col];
                    _inputGates[gate] = input;
                    _hiddenGates[gate] = recurrent;
                }
                for (int col = 0; col < Hidden; col++)
                {
                    float reset = Sigmoid(_inputGates[col] + _hiddenGates[col]);
                    float keep = Sigmoid(_inputGates[Hidden + col] + _hiddenGates[Hidden + col]);
                    float candidate = (float)Math.Tanh(_inputGates[2 * Hidden + col] + reset * _hiddenGates[2 * Hidden + col]);
                    _hidden[col] = (1 - keep) * candidate + keep * _hidden[col];
                }
            }
            float sum = 0;
            for (int row = 0; row < Options; row++)
            {
                float value = _weights[Output + Options * Hidden + row];
                for (int col = 0; col < Hidden; col++)
                    value += _weights[Output + row * Hidden + col] * _hidden[col];
                _values[row] = value;
                sum += value;
            }
            for (int row = 0; row < Options; row++) _values[row] -= sum / Options;
        }
    }
}
