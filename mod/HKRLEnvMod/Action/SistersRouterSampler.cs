using System;
using System.Globalization;
using System.IO;

namespace HKRLEnvMod.Action
{
    internal sealed class SistersRouterSampler : IDisposable
    {
        private readonly Random _rng;
        private readonly BinaryWriter _log;
        private readonly double _temperature;
        internal SistersRouterSampler(string folder, ulong episode, byte[] digest)
        {
            if (!int.TryParse(Environment.GetEnvironmentVariable("HKRL_NATIVE_DEMO_SEED"), out var seed) ||
                !double.TryParse(Environment.GetEnvironmentVariable("HKRL_NATIVE_ROUTER_TEMPERATURE"), NumberStyles.Float,
                    CultureInfo.InvariantCulture, out _temperature) || !(_temperature > 0 && _temperature < 1000))
                throw new InvalidOperationException("Missing router rollout seed or temperature");
            _rng = new Random(seed);
            _log = new BinaryWriter(new FileStream(Path.Combine(folder,episode+".routes"),FileMode.CreateNew,FileAccess.Write,FileShare.Read));
            _log.Write(0x31525453); _log.Write(episode); _log.Write(digest); _log.Write(seed); _log.Write(_temperature);
        }
        internal bool Choose(float time,double score)
        {
            double p=.04+.92/(1+Math.Exp(-Math.Max(-50,Math.Min(50,score/_temperature))));
            double uniform=_rng.NextDouble();
            bool alternate=uniform<p;
            _log.Write(time); _log.Write(score); _log.Write(p); _log.Write(uniform); _log.Write(alternate);
            return alternate;
        }
        internal void Flush() => _log.Flush();
        public void Dispose() => _log.Dispose();
    }
}
