using System;
using System.IO;

namespace HKRLEnvMod.Debug
{
    /// <summary>Local, opt-in physics-tick capture. Each raw input word precedes an
    /// ordinary size-prefixed StepResponse. Physical bindings remain owned by InControl.</summary>
    internal sealed class HumanDemonstration : IDisposable
    {
        private BinaryWriter? _writer;
        private string? _target;
        internal bool Armed => _target != null;
        internal bool Recording => _writer != null;
        internal uint PreviousButtons { get; private set; }
        internal int TaskId { get; private set; }

        internal void ArmFromLocalRequest(int taskId)
        {
            string directory = Path.GetDirectoryName(typeof(HumanDemonstration).Assembly.Location)!;
            string marker = Path.Combine(directory, "hkrl-human-demo.path");
            if (!File.Exists(marker)) return;
            string target = File.ReadAllText(marker).Trim();
            if (!Path.IsPathRooted(target) || !target.EndsWith(".hkdem", StringComparison.OrdinalIgnoreCase))
                throw new IOException("Expected an absolute local .hkdem capture path");
            if (File.Exists(target) || File.Exists(target + ".partial"))
                throw new IOException("Capture path is already occupied");
            File.Delete(marker);
            _target = target;
            TaskId = taskId;
        }

        internal void Write(ushort keys, byte[] frame)
        {
            if (_writer == null)
            {
                _writer = new BinaryWriter(new BufferedStream(new FileStream(
                    _target + ".partial", FileMode.CreateNew, FileAccess.Write, FileShare.Read), 65536));
                _writer.Write(new byte[] { 72, 75, 68, 49 }); // HKD1
            }
            _writer.Write(keys);
            _writer.Write(frame);
            // Describe physical game inputs already observed, never the new label in s_t.
            PreviousButtons = ((keys & 16) != 0 ? 2u : 0u) | ((keys & 32) != 0 ? 4u : 0u)
                | ((keys & 64) != 0 ? 8u : 0u) | ((keys & 128) != 0 ? 16u : 0u)
                | ((keys & 256) != 0 ? 32u : 0u) | ((keys & 512) != 0 ? 64u : 0u);
        }

        internal static ushort PhysicalKeys()
        {
            var a = InputHandler.Instance?.inputActions;
            if (a == null) return 0;
            return (ushort)((a.left.IsPressed ? 1 : 0) | (a.right.IsPressed ? 2 : 0)
                | (a.down.IsPressed ? 4 : 0) | (a.up.IsPressed ? 8 : 0)
                | (a.jump.IsPressed ? 16 : 0) | (a.dash.IsPressed ? 32 : 0)
                | (a.attack.IsPressed ? 64 : 0) | (a.quickCast.IsPressed ? 128 : 0)
                | (a.cast.IsPressed ? 256 : 0) | (a.dreamNail.IsPressed ? 512 : 0));
        }

        internal void Finish(bool complete)
        {
            string? target = _target;
            bool wrote = Recording;
            _target = null;
            PreviousButtons = 0;
            try { _writer?.Dispose(); }
            finally { _writer = null; }
            if (complete && wrote && target != null) File.Move(target + ".partial", target);
        }

        public void Dispose() => Finish(false);
    }
}
