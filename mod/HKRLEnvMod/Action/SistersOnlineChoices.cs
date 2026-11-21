using System;
using System.IO;

namespace HKRLEnvMod.Action
{
    // Rare randomized actions in developer games, followed by the frozen policy.
    // The time between trials exceeds the learner's outcome measurement window.
    internal sealed class SistersOnlineChoices : IDisposable
    {
        private readonly string _directory;
        private readonly Random _random;
        private BinaryWriter? _writer;
        private float _next, _release;
        private int _movement;
        private readonly bool _descent = Environment.GetEnvironmentVariable("HKRL_NATIVE_DESCENT_TRIAL") == "1";

        internal SistersOnlineChoices(string directory, int seed)
        {
            _directory = directory;
            _random = new Random(seed);
        }

        internal void Reset(ulong episode)
        {
            _writer?.Dispose();
            _writer = new BinaryWriter(new FileStream(Path.Combine(_directory, episode + ".choices"),
                FileMode.CreateNew, FileAccess.Write, FileShare.Read));
            _writer.Write(_descent ? 0x32434B48 : 0x31434B48);
            _writer.Write(episode);
            _next = _release = 0;
        }

        private static bool NearDescent(float[] x)
        {
            if (x[9] > .5f || x[SistersTimingState.TrioFlag] < .5f) return false;
            for (int k = SistersTimingState.Hero; k < SistersTimingState.Hero + 3 * SistersTimingState.Boss; k += SistersTimingState.Boss)
                if (x[k] > .5f && Math.Abs(x[k + 1]) <= .4f &&
                    (x[k + 20] > .5f || x[k + 21] > .5f || x[k + 22] > .5f)) return true;
            return false;
        }

        internal PrimitiveInput Choose(float time, NativeSistersNetwork policy, PrimitiveInput greedy, float[] state)
        {
            if (_descent && time < _release) return policy.BestHorizontal(_movement);
            if (time < _next) return greedy;
            if (_descent && !NearDescent(state)) return greedy;
            if (!policy.MovementPair(out int first, out int other, out float gap, _descent) || gap > 3f)
            {
                _next = time + .08f;
                return greedy;
            }
            int draw = _random.Next(2);
            _next = time + .72f + .24f * (float)_random.NextDouble();
            if (_descent)
            {
                _movement = (draw == 0 ? greedy : policy.Command(other)).MovementX;
                _release = time + .16f;
                _next = time + 1.5f;
            }
            var writer = _writer ?? throw new InvalidOperationException("Development episode not open");
            writer.Write(time);
            writer.Write(first);
            writer.Write(other);
            writer.Write(draw);
            writer.Write(.5f);
            writer.Write(gap);
            return draw == 0 ? greedy : policy.Command(other);
        }

        internal void Flush() => _writer?.Flush();
        public void Dispose() => _writer?.Dispose();
    }
}
