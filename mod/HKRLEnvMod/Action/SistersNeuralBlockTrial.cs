using System;
using System.IO;
using System.Security.Cryptography;

namespace HKRLEnvMod.Action
{
    // Development-only random assignment between two frozen neural policies.
    // Both histories see the same observations, while only the acting network runs inference.
    internal sealed class SistersNeuralBlockTrial : IDisposable
    {
        private readonly string _folder, _candidate;
        private readonly Random _random;
        private NativeSistersNetwork? _alternate;
        private BinaryWriter? _log;
        private float _until;
        private bool _candidateBlock;

        internal SistersNeuralBlockTrial(string folder, string candidate, int seed)
        {
            _folder = folder;
            _candidate = candidate;
            _random = new Random(seed);
        }

        internal void Reset(ulong episode, byte[] parent)
        {
            var candidate = File.ReadAllBytes(_candidate);
            _ = NativeSistersNetwork.Load(candidate, out _alternate);
            if (_alternate == null || parent.Length < 12 || BitConverter.ToInt32(parent, 0) != 0x334C5149)
                throw new InvalidDataException("The comparison requires two phase policies");
            int openingBytes = BitConverter.ToInt32(parent, 4);
            if (openingBytes != BitConverter.ToInt32(candidate, 4))
                throw new InvalidDataException("Different neural openings");
            for (int i = 12; i < 12 + openingBytes; i++)
                if (parent[i] != candidate[i]) throw new InvalidDataException("Different neural openings");
            _log?.Dispose();
            _log = new BinaryWriter(new FileStream(Path.Combine(_folder, episode + ".arms"),
                FileMode.CreateNew, FileAccess.Write, FileShare.Read));
            _log.Write(0x314D5241);
            _log.Write(episode);
            using var sha = SHA256.Create();
            _log.Write(sha.ComputeHash(parent));
            _log.Write(sha.ComputeHash(candidate));
            _until = -1;
        }

        internal NativeSistersNetwork Select(NativeSistersNetwork active, bool trio, float[] state, float time)
        {
            if (time >= _until)
            {
                _candidateBlock = _random.NextDouble() < .5;
                _until = time + .8f;
                _log!.Write(time);
                _log.Write((byte)(_candidateBlock ? 1 : 0));
            }
            if (trio && _candidateBlock)
            {
                active.Remember(state, time);
                return _alternate!;
            }
            _alternate!.Remember(state, time);
            return active;
        }

        internal void Flush() => _log?.Flush();
        public void Dispose() => _log?.Dispose();
    }
}
