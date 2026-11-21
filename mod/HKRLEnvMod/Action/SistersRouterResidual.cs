using System;
using System.IO;

namespace HKRLEnvMod.Action
{
    // The existing router still supplies its original logit. Learn a bounded
    // change using the same block-start observation and its projected features.
    internal sealed class SistersRouterResidual
    {
        internal const int Magic = 0x3B4C5149;
        private static readonly int[] Raw = { 0,1,2,3,4,5,7,8,9,13,19,25,26,27,28,34,61,62,97,98,132,133,134,135,146,147,148,149 };
        private readonly float[] _x = new float[40], _h = new float[32], _z = new float[16];
        private readonly float[] _w1, _b1, _w2, _b2, _w3, _b3;
        private readonly float _bound;

        internal SistersRouterResidual(BinaryReader reader)
        {
            _bound = reader.ReadSingle();
            if (!(_bound > 0 && _bound < 1000)) throw new InvalidDataException("Invalid router residual bound");
            _w1 = NativeSistersNetwork.ReadWeights(reader, 40*32);
            _b1 = NativeSistersNetwork.ReadWeights(reader, 32);
            _w2 = NativeSistersNetwork.ReadWeights(reader, 32*16);
            _b2 = NativeSistersNetwork.ReadWeights(reader, 16);
            _w3 = NativeSistersNetwork.ReadWeights(reader, 16);
            _b3 = NativeSistersNetwork.ReadWeights(reader, 1);
            if (reader.BaseStream.Position != reader.BaseStream.Length) throw new InvalidDataException("Unexpected router weights");
        }

        internal void SetParentFeature(int index, double feature) => _x[index] = (float)(feature/3);

        private static void Relu(float[] x, float[] weights, float[] bias, float[] output)
        {
            for (int i = 0; i < output.Length; i++)
            {
                double v = bias[i];
                for (int j = 0; j < x.Length; j++) v += (double)weights[i*x.Length+j] * x[j];
                output[i] = (float)Math.Max(0, v);
            }
        }

        internal double Change(float[] state)
        {
            for (int i = 0; i < Raw.Length; i++) _x[i+12] = state[Raw[i]];
            Relu(_x,_w1,_b1,_h);
            Relu(_h,_w2,_b2,_z);
            double v = _b3[0];
            for (int j=0; j<_z.Length; j++) v += (double)_w3[j]*_z[j];
            return _bound * Math.Tanh(v);
        }
    }
}
