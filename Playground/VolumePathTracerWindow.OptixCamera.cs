using System;
using System.Numerics;

namespace Playground
{

    public class OptixCamera
    {
        public Vector3 mPosition;
        public Vector3 mForward;
        public Vector2 mResolution;
        public Matrix4x4 mRasterToWorld;
        public Matrix4x4 mWorldToRaster;
        public float mImagePlaneDist;

        public void Setup( 
            Vector3 aPosition,
            Vector3 aForward,
            Vector3 aUp,
            Vector2 aResolution,
            float aHorizontalFOV)
        {

            var forward = Vector3.Normalize(aForward);
            var up = Vector3.Normalize(Vector3.Cross(aUp, -forward));
            var left = -Vector3.Cross(-forward, up);

            mPosition = aPosition;
            mForward = forward;
            mResolution = aResolution;

            var pos = new Vector3(
                Vector3.Dot(up, aPosition),
                Vector3.Dot(left, aPosition),
                Vector3.Dot(-forward, aPosition));

            var worldToCamera = Matrix4x4.Identity;
            SetRow(ref worldToCamera, 0, new Vector4(up, -pos.X));
            SetRow(ref worldToCamera, 1, new Vector4(left, -pos.Y));
            SetRow(ref worldToCamera, 2, new Vector4(-forward, -pos.Z));

            var perspective = Perspective(aHorizontalFOV *MathF.PI / 180.0f,  0.1f, 10000f, aResolution.X / aResolution.Y);
            var worldToNScreen = perspective * worldToCamera;
            Matrix4x4.Invert(worldToNScreen, out var nscreenToWorld);
                

            mWorldToRaster =
                Matrix4x4.CreateScale(new Vector3(aResolution.X * 0.5f, aResolution.Y * 0.5f, 0)) *
                Matrix4x4.CreateTranslation(new Vector3(1f, 1f, 0)) * worldToNScreen;

            mRasterToWorld = nscreenToWorld *
                             Matrix4x4.CreateTranslation(new Vector3(-1f, -1f, 0)) *
                             Matrix4x4.CreateScale(new Vector3(2f / aResolution.X, 2f / aResolution.Y, 0));

            var tanHalfAngle = MathF.Tan(aHorizontalFOV *MathF.PI / 360f);
            mImagePlaneDist = aResolution.X / (2f * tanHalfAngle);
        }

        private Matrix4x4 Perspective(
            float aFov,
            float aNear,
            float aFar,
            float aspect)
        {
            // Camera points towards -z.  0 < near < far.
            // Matrix maps z range [-near, -far] to [-1, 1], after homogeneous division.
            float f = 1f / (MathF.Tan(aFov * MathF.PI / 360.0f));
            float d = 1f / (aNear - aFar);

            Matrix4x4 r;
            r.M11 = f / aspect; r.M12 = 0.0f; r.M13 = 0.0f; r.M14 = 0.0f;
            r.M21 = 0.0f; r.M22 = -f; r.M23 = 0.0f; r.M24 = 0.0f;
            r.M31 = 0.0f; r.M32 = 0.0f; r.M33 = (aNear + aFar) * d; r.M34 = 2.0f * aNear * aFar * d;
            r.M41 = 0.0f; r.M42 = 0.0f; r.M43 = -1.0f; r.M44 = 0.0f;

            return r;
        }

        private void SetRow(ref Matrix4x4 m, int index, Vector4 row)
        {
            switch (index)
            {
                case 0:
                    m.M11 = row.X;
                    m.M12 = row.Y;
                    m.M13 = row.Z;
                    m.M14 = row.W;
                    break;
                case 1:
                    m.M21 = row.X;
                    m.M22 = row.Y;
                    m.M23 = row.Z;
                    m.M24 = row.W;
                    break;
                case 2:
                    m.M31 = row.X;
                    m.M32 = row.Y;
                    m.M33 = row.Z;
                    m.M34 = row.W;
                    break;
                case 3:
                    m.M41 = row.X;
                    m.M42 = row.Y;
                    m.M43 = row.Z;
                    m.M44 = row.W;
                    break;
            }
        }
    }
}