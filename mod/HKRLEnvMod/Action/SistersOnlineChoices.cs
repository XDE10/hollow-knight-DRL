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
        private readonly SistersCompoundOption? _compound;
        private readonly SistersDirectionalOptions? _directional;
        private readonly bool _descent = Environment.GetEnvironmentVariable("HKRL_NATIVE_DESCENT_TRIAL") == "1";

        internal SistersOnlineChoices(string directory, int seed)
        {
            _directory = directory;
            _random = new Random(seed);
            var riskPath = Environment.GetEnvironmentVariable("HKRL_NATIVE_COMPOUND_RISK");
            var sampledHead = Environment.GetEnvironmentVariable("HKRL_NATIVE_OPTION_SAMPLER");
            var directional = Environment.GetEnvironmentVariable("HKRL_NATIVE_DIRECTIONAL_SAMPLER");
            if (!string.IsNullOrEmpty(directional))
            {
                if (_descent || !string.IsNullOrEmpty(sampledHead) || !string.IsNullOrEmpty(riskPath))
                    throw new InvalidOperationException("Choose one native sampler");
                _directional = new SistersDirectionalOptions(File.ReadAllBytes(directional!));
            }
            else if (!string.IsNullOrEmpty(sampledHead))
            {
                if (_descent || !string.IsNullOrEmpty(riskPath)) throw new InvalidOperationException("Choose one option sampler");
                _compound = SistersCompoundOption.SamplePolicy(sampledHead!);
            }
            else if (!string.IsNullOrEmpty(riskPath))
            {
                if (_descent) throw new InvalidOperationException("Choose one randomized experiment");
                _compound = SistersCompoundOption.Training(riskPath!);
            }
        }

        internal void Reset(ulong episode)
        {
            _writer?.Dispose();
            _writer = new BinaryWriter(new FileStream(Path.Combine(_directory, episode + ".choices"),
                FileMode.CreateNew, FileAccess.Write, FileShare.Read));
            _compound?.Reset();
            _directional?.Reset();
            _writer.Write(_directional != null ? SistersDirectionalOptions.TapeMagic : _compound?.LearnedSampling == true ? 0x34434B48 : _compound != null ? 0x33434B48 : _descent ? 0x32434B48 : 0x31434B48);
            _writer.Write(episode);
            if (_directional != null)
            {
                _writer.Write(_directional.Interval);
                _writer.Write(_directional.Digest);
            }
            else if (_compound?.LearnedSampling == true)
            {
                _writer.Write(_compound.Interval);
                _writer.Write(_compound.SamplingDigest!);
            }
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
            if (_directional != null) return _directional.Choose(policy, state, time, _random, _writer);
            if (_compound != null) return _compound.Choose(policy, state, time, _random, _writer);
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
