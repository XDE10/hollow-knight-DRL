using System;
using System.IO;

namespace HKRLEnvMod.Action
{
    // Teacher-free native Categorical draws. The Python learner reconstructs this
    // exact distribution from the frozen checkpoint and the same observations.
    internal sealed class SistersPpoSampler : IDisposable
    {
        internal const string Flag = "HKRL_NATIVE_PPO_ROLLOUT";
        private readonly BinaryWriter _tape;
        private readonly double _temperature;
        private readonly double[] _weights;
        private ulong _random;

        internal SistersPpoSampler(string folder, ulong episode, byte[] digest,
            int seed, double temperature, int actions)
        {
            if (!(temperature > 0) || double.IsInfinity(temperature))
                throw new ArgumentException("Invalid PPO temperature");
            _random = unchecked((ulong)(uint)seed ^ (episode * 0x9E3779B97F4A7C15UL));
            _temperature = temperature;
            _weights = new double[actions];
            _tape = new BinaryWriter(new FileStream(Path.Combine(folder, episode + ".ppo"),
                FileMode.CreateNew, FileAccess.Write, FileShare.Read, 65536));
            _tape.Write(0x314F5050);
            _tape.Write(episode);
            _tape.Write(digest);
            _tape.Write(seed);
            _tape.Write(temperature);
        }

        private double Draw()
        {
            unchecked
            {
                _random += 0x9E3779B97F4A7C15UL;
                ulong x = _random;
                x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
                x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
                return ((x ^ (x >> 31)) >> 11) * (1.0 / 9007199254740992.0);
            }
        }

        internal PrimitiveInput Choose(NativeSistersNetwork actor, float time)
        {
            double maximum = actor.Score(actor.BestIndex);
            double total = 0;
            for (int i = 0; i < _weights.Length; i++)
            {
                _weights[i] = Math.Exp((actor.Score(i) - maximum) / _temperature);
                total += _weights[i];
            }
            if (!(total >= 1) || double.IsInfinity(total))
                throw new InvalidDataException("Nonfinite PPO actor scores");
            double uniform = Draw();
            double remaining = uniform * total;
            int selected = 0;
            for (int i = 0; i < _weights.Length; i++)
            {
                if (_weights[i] > 0) selected = i;
                remaining -= _weights[i];
                if (remaining < 0) break;
            }
            double logProbability = (actor.Score(selected) - maximum) / _temperature - Math.Log(total);
            _tape.Write(time);
            _tape.Write(selected);
            _tape.Write(uniform);
            _tape.Write(logProbability);
            return actor.Command(selected);
        }

        internal void Flush() => _tape.Flush();
        public void Dispose() => _tape.Dispose();
    }
}
