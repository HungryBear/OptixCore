using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using OptixCore.Library.Native;

namespace OptixCore.Library.Scene
{
    public class OptixOBJLoader : OBJLoader
    {
        public string IntersecitonProgPath { get; set; }
        public string IntersecitonProgName { get; set; }
        public string BoundingBoxProgPath { get; set; }
        public string BoundingBoxProgName { get; set; }

        public Context Context { get; set; }
        public Material Material { get; set; }
        public GeometryGroup GeoGroup { get; set; }

        public AccelBuilder Builder { get; set; }
        public AccelTraverser Traverser { get; set; }

        private bool dataLoaded;

        public OptixOBJLoader(string filename, Context context, GeometryGroup geoGroup, Material material)
            : base(filename)
        {
            Context = context;
            GeoGroup = geoGroup;
            Material = material;

            Builder = AccelBuilder.Sbvh;
            Traverser = AccelTraverser.Bvh;
        }

        public void LoadObjData()
        {
            this.LoadObj();
            dataLoaded = true;
        }

        public override void LoadContent(Func<string, string, Material> materialResolver = null)
        {
            //float start = Time.GetTimeInSecs();
            if (!dataLoaded)
            {
                LoadObj();
            }

            if (materialResolver == null)
                CreateGeometry();
            else
                CreateGeometry(materialResolver);

            //Trace.WriteLine(string.Format("{0:0.00}s", Time.GetTimeInSecs() - start));

        }


        protected override void CreateGeometry()
        {
            base.CreateGeometry();

            if (!GenerateGeometry)
                return;

            //create buffer descriptions
            var vDesc = new BufferDesc() { Width = (uint)Positions.Count, Format = Format.Float3, Type = BufferType.Input };
            var nDesc = new BufferDesc() { Width = (uint)Normals.Count, Format = Format.Float3, Type = BufferType.Input };
            var tcDesc = new BufferDesc() { Width = (uint)Texcoords.Count, Format = Format.Float2, Type = BufferType.Input };

            // Create the buffers to hold our geometry data
            var vBuffer = new OptixBuffer(Context, vDesc);
            var nBuffer = new OptixBuffer(Context, nDesc);
            var tcBuffer = new OptixBuffer(Context, tcDesc);

            vBuffer.SetData<Vector3>(Positions.ToArray());
            nBuffer.SetData<Vector3>(Normals.ToArray());
            tcBuffer.SetData<Vector2>(Texcoords.ToArray());

            List<GeometryInstance> instances = new List<GeometryInstance>();
            foreach (ObjGroup group in Groups)
            {
                //empty group
                if (group.VIndices.Count == 0 && group.NIndices.Count == 0 && group.TIndices.Count == 0)
                    continue;

                //ValidateGroup( group );

                var normalsUseVIndices = GenerateNormals && group.NIndices.Count == 0 && Normals.Count > 0;

                if (normalsUseVIndices)
                    Debug.Assert(Normals.Count == Positions.Count);

                var numNormIndices = normalsUseVIndices ? group.VIndices.Count : group.NIndices.Count;

                var viDesc = new BufferDesc { Width = (uint)group.VIndices.Count, Format = Format.Int3, Type = BufferType.Input };
                var niDesc = new BufferDesc { Width = (uint)numNormIndices, Format = Format.Int3, Type = BufferType.Input };
                var tiDesc = new BufferDesc { Width = (uint)group.TIndices.Count, Format = Format.Int3, Type = BufferType.Input };

                var viBuffer = new OptixBuffer(Context, viDesc);
                var niBuffer = new OptixBuffer(Context, niDesc);
                var tiBuffer = new OptixBuffer(Context, tiDesc);

                viBuffer.SetData(group.VIndices.ToArray());
                //if normals weren't in the obj and we genereated them, use the vertex indices
                niBuffer.SetData(normalsUseVIndices ? group.VIndices.ToArray() : group.NIndices.ToArray());
                tiBuffer.SetData(group.TIndices.ToArray());

                //create a geometry node and set the buffers
                Geometry geometry = new Geometry(Context);
                geometry.IntersectionProgram = new OptixProgram(Context, IntersecitonProgPath, IntersecitonProgName);
                geometry.BoundingBoxProgram = new OptixProgram(Context, BoundingBoxProgPath, BoundingBoxProgName);
                geometry.PrimitiveCount = (uint)group.VIndices.Count;

                geometry["vertex_buffer"].Set(vBuffer);
                geometry["normal_buffer"].Set(nBuffer);
                geometry["texcoord_buffer"].Set(tcBuffer);
                geometry["vindex_buffer"].Set(viBuffer);
                geometry["nindex_buffer"].Set(niBuffer);
                geometry["tindex_buffer"].Set(tiBuffer);

                //create a geometry instance
                GeometryInstance instance = new GeometryInstance(Context);
                instance.Geometry = geometry;
                instance.AddMaterial(Material);

                if (group.mtrl != null)
                {
                    ObjMaterial mtrl = mMtrls[group.mtrl];
                    instance["diffuse_color"].SetFloat3(ref mtrl.Kd);
                    instance["emission_color"].SetFloat3(ref mtrl.Ke);
                }
                else
                {
                    instance["diffuse_color"].Set(1.0f, 1.0f, 1.0f);
                }

                instances.Add(instance);
            }

            //create an acceleration structure for the geometry
            Acceleration accel = new Acceleration(Context, Builder, Traverser);

            if (Builder == AccelBuilder.Sbvh || Builder == AccelBuilder.TriangleKdTree)
            {
                accel.VertexBufferName = "vertex_buffer";
                accel.IndexBufferName = "vindex_buffer";
            }

            //now attach the instance and accel to the geometry group
            GeoGroup.Acceleration = accel;
            GeoGroup.AddChildren(instances);
        }

        protected override void CreateGeometry(Func<string, string, Material> materialResolver)
        {
            if (!GenerateGeometry)
                return;

            //create buffer descriptions
            BufferDesc vDesc = new BufferDesc() { Width = (uint)Positions.Count, Format = Format.Float3, Type = BufferType.Input };
            BufferDesc nDesc = new BufferDesc() { Width = (uint)Normals.Count, Format = Format.Float3, Type = BufferType.Input };
            BufferDesc tcDesc = new BufferDesc() { Width = (uint)Texcoords.Count, Format = Format.Float2, Type = BufferType.Input };

            // Create the buffers to hold our geometry data
            var vBuffer = new OptixBuffer(Context, vDesc);
            var nBuffer = new OptixBuffer(Context, nDesc);
            var tcBuffer = new OptixBuffer(Context, tcDesc);

            vBuffer.SetData<Vector3>(Positions.ToArray());
            nBuffer.SetData<Vector3>(Normals.ToArray());
            tcBuffer.SetData<Vector2>(Texcoords.ToArray());

            List<GeometryInstance> instances = new List<GeometryInstance>();
            foreach (ObjGroup group in Groups)
            {
                //empty group
                if (group.VIndices.Count == 0 && group.NIndices.Count == 0 && group.TIndices.Count == 0)
                    continue;

                ValidateGroup(group);

                bool normalsUseVIndices = GenerateNormals && group.NIndices.Count == 0 && Normals.Count > 0;

                if (normalsUseVIndices)
                    System.Diagnostics.Debug.Assert(Normals.Count == Positions.Count);

                int numNormIndices = normalsUseVIndices ? group.VIndices.Count : group.NIndices.Count;

                var viDesc = new BufferDesc() { Width = (uint)group.VIndices.Count, Format = Format.Int3, Type = BufferType.Input };
                var niDesc = new BufferDesc() { Width = (uint)numNormIndices, Format = Format.Int3, Type = BufferType.Input };
                var tiDesc = new BufferDesc() { Width = (uint)group.TIndices.Count, Format = Format.Int3, Type = BufferType.Input };

                var viBuffer = new OptixBuffer(Context, viDesc);
                var niBuffer = new OptixBuffer(Context, niDesc);
                var tiBuffer = new OptixBuffer(Context, tiDesc);

                viBuffer.SetData<Int3>(group.VIndices.ToArray());
                //if normals weren't in the obj and we genereated them, use the vertex indices
                niBuffer.SetData<Int3>(normalsUseVIndices ? group.VIndices.ToArray() : group.NIndices.ToArray());
                tiBuffer.SetData<Int3>(group.TIndices.ToArray());

                //create a geometry node and set the buffers
                var geometry = new Geometry(Context);
                geometry.IntersectionProgram = new OptixProgram(Context, IntersecitonProgPath, IntersecitonProgName);
                geometry.BoundingBoxProgram = new  OptixProgram(Context, BoundingBoxProgPath, BoundingBoxProgName);
                geometry.PrimitiveCount = (uint)group.VIndices.Count;

                geometry["vertex_buffer"].Set(vBuffer);
                geometry["normal_buffer"].Set(nBuffer);
                geometry["texcoord_buffer"].Set(tcBuffer);
                geometry["vindex_buffer"].Set(viBuffer);
                geometry["nindex_buffer"].Set(niBuffer);
                geometry["tindex_buffer"].Set(tiBuffer);

                //create a geometry instance
                GeometryInstance instance = new GeometryInstance(Context);
                instance.Geometry = geometry;
                instance.AddMaterial(materialResolver(group.name, group.mtrl));

                if (group.mtrl != null)
                {
                    var mtrl = mMtrls[group.mtrl];
                    instance["diffuse_color"].SetFloat3(ref mtrl.Kd);
                    instance["specular_color"].SetFloat3(ref mtrl.Ks);
                    //instance["emission_color"].SetFloat3(mtrl.Ke);
                }
                else
                {
                    instance["diffuse_color"].Set(1.0f, 1.0f, 1.0f);
                }

                instances.Add(instance);
            }

            //create an acceleration structure for the geometry
            Acceleration accel = new Acceleration(Context, Builder, Traverser);

            if (Builder == AccelBuilder.Sbvh || Builder == AccelBuilder.TriangleKdTree)
            {
                accel.VertexBufferName = "vertex_buffer";
                accel.IndexBufferName = "vindex_buffer";
            }

            //now attach the instance and accel to the geometry group
            GeoGroup.Acceleration = accel;
            GeoGroup.AddChildren(instances);
        }
    }
}