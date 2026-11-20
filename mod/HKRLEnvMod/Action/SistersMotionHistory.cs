using System;

namespace HKRLEnvMod.Action
{
    // The native actor accepts at most one decision per two physics ticks.
    // This buffer spans both lags even at the game's full 50 Hz physics rate.
    internal sealed class SistersMotionHistory
    {
        private const int Capacity = 64, Width = SistersTimingState.Width, Hidden = 32;
        private static readonly double[] Lags = { .16, .48 };
        private readonly float[] _states = new float[Capacity * Width];
        private readonly float[] _times = new float[Capacity];
        private readonly float[] _weights, _bias;
        private int _next, _count;

        internal SistersMotionHistory(float[] weights, float[] bias)
        {
            _weights = weights;
            _bias = bias;
        }

        private void CheckContinuity(float now)
        {
            if (_count == 0) return;
            double elapsed = (double)now - _times[(_next + Capacity - 1) % Capacity];
            if (elapsed < 0 || elapsed > .24001) _count = 0;
        }

        internal void Remember(float[] raw, float now)
        {
            CheckContinuity(now);
            int start = _next * Width;
            Array.Copy(raw, 0, _states, start, Width);
            for (int slot = 0; slot < 5; slot++)
            {
                int k = slot < 3 ? 24 + slot * 36 : 132 + (slot - 3) * 14;
                if (raw[k] == 0) Array.Clear(_states, start + k, slot < 3 ? 36 : 14);
            }
            _times[_next] = now;
            _next = (_next + 1) % Capacity;
            _count = Math.Min(Capacity, _count + 1);
        }

        internal void Encode(float now, float[] output, int offset)
        {
            CheckContinuity(now);
            Array.Clear(output, offset, Hidden * Lags.Length);
            foreach (double lag in Lags)
            {
                for (int past = 1; past <= _count; past++)
                {
                    int index = (_next + Capacity - past) % Capacity;
                    double age = (double)now - _times[index];
                    if (age < lag - .00001) continue;
                    if (age <= lag + .24001)
                    {
                        int source = index * Width;
                        for (int j = 0; j < Hidden; j++)
                        {
                            float value = _bias[j];
                            int w = j * (Width + 1);
                            for (int k = 0; k < Width; k++)
                                value += _states[source + k] * _weights[w + k];
                            output[offset + j] = Math.Max(0f, value + (float)age * _weights[w + Width]);
                        }
                    }
                    break;
                }
                offset += Hidden;
            }
        }
    }
}
