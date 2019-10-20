using System;
using System.Numerics;

namespace OptixCore.Library.Scene
{
    public interface ICamera
    {
        Vector3 Position { get; set; }
        Vector3 Target { get; set; }
        Vector3 Right { get; set; }
        Vector3 Up { get; set; }
        Vector3 Look { get; set; }

        float Aspect { get; set; }
        float Fov { get; set; }

        float TranslationVel { get; set; }
        float RotationVel { get; set; }

        void Dolly(float x);
        void Pan(float x, float y);
        void Rotate(float x, float y);

        void LookAt(Vector3 pos, Vector3 target, Vector3 up);
        void CenterOnBoundingBox(BoundingBox box);
        void CenterOnBoundingBox(BoundingBox box, float scale);
        void CenterOnBoundingBox(BoundingBox box, Vector3 up, int axis, float scale);
    }

    public class Camera : ICamera
    {
        protected Vector3 mRight;
        protected Vector3 mUp;
        protected Vector3 mLook;

        protected float mUlength = 0.0f;
        protected float mVlength = 0.0f;

        public Camera()
        {
            TranslationVel = 1.0f;
            RotationVel = 1.0f;

            Aspect = 1.0f;
            Fov = 60.0f;

            Position = Vector3.Zero;
            mRight = Vector3.UnitX;
            mUp = Vector3.UnitY;
            mLook = Vector3.UnitZ;
        }

        public void Yaw(float angle)
        {
            Matrix4x4 rot = Matrix4x4.CreateRotationY(angle);

            mRight = Vector3.TransformNormal(mRight, rot);
            mLook = Vector3.TransformNormal(mLook, rot);
        }

        public void Pitch(float angle)
        {
            Matrix4x4 rot = Matrix4x4.CreateFromAxisAngle(mRight, angle);

            mUp = Vector3.TransformNormal(mUp, rot);
            mLook = Vector3.TransformNormal(mLook, rot);
        }

        #region ICamera Members

        public Vector3 Position { get; set; }
        public Vector3 Target { get; set; }
        public Vector3 Right { get { return mRight * mUlength; } set { mRight = value; } }
        public Vector3 Up { get { return mUp * mVlength; } set { mUp = value; } }
        public Vector3 Look { get { return mLook; } set { mLook = value; } }

        public float Aspect { get; set; }
        public float Fov { get; set; }

        public float TranslationVel { get; set; }
        public float RotationVel { get; set; }

        public virtual void Dolly(float dt)
        {
            Position += mLook * (TranslationVel * dt);
        }

        public virtual void Pan(float dx, float dy)
        {
            dx *= TranslationVel;
            dy *= TranslationVel;

            Position += new Vector3(mRight.X * dx, 0.0f, mRight.Z * dx);
            Position += mUp * dy;
        }

        public virtual void Rotate(float dx, float dy)
        {
            Yaw(dx * RotationVel);
            Pitch(dy * RotationVel);
            BuildView();
        }

        public virtual void LookAt(Vector3 pos, Vector3 target, Vector3 up)
        {
            var _up = Vector3.Normalize(up);

            Vector3 L = target - pos;
            mLook = L;

            L = Vector3.Normalize(L);

            mRight = Vector3.Cross(L, _up);

            Position = pos;
            Target = target;

            BuildView();
        }

        public virtual void BuildView()
        {
            float wlen = mLook.Length();
            mUlength = wlen * (float)Math.Tan(Fov / 2.0f * Math.PI / 180.0f);
            mVlength = mUlength / Aspect;

            mLook = Vector3.Normalize(mLook);

            mUp = Vector3.Cross(mRight, mLook);
            mUp = Vector3.Normalize(mUp);

            mRight = Vector3.Normalize(mRight);
        }

        public virtual void CenterOnBoundingBox(BoundingBox box)
        {
            Vector3 eye = box.Center;
            Vector3 target = box.Center;
            Vector3 up = new Vector3(0.0f, 1.0f, 0.0f);

            eye.Z += 2.0f * (box.Max - box.Min).Length();
            LookAt(eye, target, up);
        }

        public virtual void CenterOnBoundingBox(BoundingBox box, float scale)
        {
            Vector3 eye = box.Center;
            Vector3 target = box.Center;
            Vector3 up = new Vector3(0.0f, 1.0f, 0.0f);

            eye.Z += 2.0f * (box.Max - box.Min).Length() * scale;
            LookAt(eye, target, up);
        }

        public virtual void CenterOnBoundingBox(BoundingBox box, Vector3 up, int axis, float scale)
        {
            Vector3 eye = box.Center;
            Vector3 target = box.Center;

            eye.Add(axis, 2.0f * (box.Max - box.Min).Length() * scale);
            LookAt(eye, target, up);
        }

        #endregion
    }
}