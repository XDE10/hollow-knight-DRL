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
        private float _next;

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
            _writer.Write(0x31434B48);
            _writer.Write(episode);
            _next = 0;
        }

        internal PrimitiveInput Choose(float time, NativeSistersNetwork policy, PrimitiveInput greedy)
        {
            if (time < _next) return greedy;
            if (!policy.MovementPair(out int first, out int other, out float gap) || gap > 3f)
            {
                _next = time + .08f;
                return greedy;
            }
            int draw = _random.Next(2);
            _next = time + .72f + .24f * (float)_random.NextDouble();
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
