using System;
using System.Globalization;
using System.IO;

namespace HKRLEnvMod.Action
{
    // Development only. The frozen evaluation uses the ordinary timing actor.
    internal sealed class SistersPrimitiveSampler : IDisposable
    {
        private readonly BinaryWriter _writer;
        private readonly Random _random;
        private readonly double _temperature;
        private double[] _mass = Array.Empty<double>();
        private float _next;

        internal SistersPrimitiveSampler(string folder, ulong episode, byte[] hash)
        {
            if (!int.TryParse(Environment.GetEnvironmentVariable("HKRL_NATIVE_DEMO_SEED"), out int seed) ||
                !double.TryParse(Environment.GetEnvironmentVariable("HKRL_NATIVE_ROUTER_TEMPERATURE"),
                    NumberStyles.Float, CultureInfo.InvariantCulture, out _temperature) || !(_temperature > 0 && _temperature < 1000))
                throw new InvalidOperationException("Missing primitive rollout parameters");
            _random = new Random(seed);
            _writer = new BinaryWriter(new FileStream(Path.Combine(folder, episode + ".keys"), FileMode.CreateNew, FileAccess.Write, FileShare.Read));
            _writer.Write(0x31534B50);
            _writer.Write(episode);
            _writer.Write(hash);
            _writer.Write(seed);
            _writer.Write(_temperature);
        }

        private static bool NearAttack(float[] state)
        {
            if (state[23] < .5f || state[9] > .5f) return false;
            for (int k = 24; k < 132; k += 36)
                if (state[k] > .5f && Math.Abs(state[k+1]) < .48f && Math.Abs(state[k+2]) < .8f) return true;
            for (int k = 132; k < 160; k += 14)
                if (state[k] > .5f && Math.Abs(state[k+1]) < .6f && Math.Abs(state[k+2]) < .42f) return true;
            return false;
        }

        internal PrimitiveInput Choose(NativeSistersNetwork actor, PrimitiveInput greedy, float[] state, float now)
        {
            if (now < _next || !NearAttack(state)) return greedy;
            _next = now + .4f;
            int n = actor.ActionCount, best = actor.BestIndex, count = 0;
            if (_mass.Length != n) _mass = new double[n];
            double sum = 0, alternativeMax = double.NegativeInfinity;
            for (int i = 0; i < n; i++)
            {
                var legal = i == best || SistersCompoundOption.Available(actor.Command(i), state);
                _mass[i] = legal ? Math.Exp(Math.Max(-50, ((double)actor.Score(i)-actor.Score(best))/_temperature)) : 0;
                if (legal) count++;
                if (legal && i != best) alternativeMax = Math.Max(alternativeMax, actor.Score(i));
                sum += _mass[i];
            }
            double alternativeSum = 0;
            for (int i = 0; i < n; i++)
                if (i != best && _mass[i] > 0)
                    alternativeSum += Math.Exp(Math.Max(-50, (actor.Score(i)-alternativeMax)/_temperature));
            double uniform = _random.NextDouble(), end = 0;
            int choice = best;
            for (int i = 0; i < n; i++)
            {
                if (_mass[i] == 0) continue;
                double alternative = i == best ? 0 : Math.Exp(Math.Max(-50, (actor.Score(i)-alternativeMax)/_temperature));
                _mass[i] = count == 1 ? 1 : .12*_mass[i]/sum + (i == best ? .78 : .08*alternative/alternativeSum + .02/(count-1));
                end += _mass[i];
                if (uniform < end) { choice = i; break; }
            }
            _writer.Write(now); _writer.Write(best); _writer.Write(choice);
            _writer.Write(uniform); _writer.Write(_mass[choice]);
            return actor.Command(choice);
        }

        internal void Flush() => _writer.Flush();
        public void Dispose() => _writer.Dispose();
    }
}
