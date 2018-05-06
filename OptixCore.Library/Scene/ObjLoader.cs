using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;


namespace OptixCore.Library.Scene
{
    public delegate void ParseHandler(string line, string[] tokens);

    public class ObjMaterial
    {
        public string Name;
        public float Ns;
        public float Ni;
        public float d;
        public float Tr;
        public float illum;

        public Vector3 Tf;
        public Vector3 Ka;
        public Vector3 Kd;
        public Vector3 Ks;
        public Vector3 Ke;

        public string AlphaTexture;
        public string BumpTexture;
        public string DiffuseTexture;
        public string SpecularTexture;
        public string GlossTexture;
        public string FresnelTexture;
        public string AmbientTexture;
    }

    public class ObjGroup
    {
        public string group;
        public string name;
        public string mtrl;
        public List<Int3> VIndices;
        public List<Int3> NIndices;
        public List<Int3> TIndices;

        public List<Vector2> TexCoords;

        public ObjGroup(string name, string group)
        {
            this.group = group;
            this.name = name;
            mtrl = null;
            VIndices = new List<Int3>();
            NIndices = new List<Int3>();
            TIndices = new List<Int3>();
            TexCoords = new List<Vector2>();
        }

        public uint[] GetIndices()
        {
            uint[] indices = new uint[VIndices.Count * 3];

            for (int i = 0; i < VIndices.Count; i++)
            {
                indices[i * 3 + 0] = (uint)VIndices[i].X;
                indices[i * 3 + 1] = (uint)VIndices[i].Y;
                indices[i * 3 + 2] = (uint)VIndices[i].Z;
            }

            return indices;
        }
    }

    public static class Vector3Extensions
    {
        public static Vector3 NormalizedCopy(this Vector3 v)
        {
            return v / v.Length();
        }
    }


    public class ObjData
    {
        public List<Vector3> Positions;
        public List<Vector3> Normals;
        public List<Vector2> Texcoords;
        public BoundingBox Bounds;
        public float BoundingSphereRadius;
        public Vector3 BoundingSphereCenter;
        public ObjGroup[] Geometry;
        public Dictionary<string, ObjMaterial> Materials;
        public string FileName;


        public void CalculateBoundingSphere()
        {
            BoundingSphereCenter = Bounds.Center;

            BoundingSphereRadius = Bounds.MaxExtent() / 2f;
        }

        public void CreateNormals()
        {

            float winding = 1.0f;

            Trace.Write("[Creating normals: ");

            for (int i = 0; i < Positions.Count; i++)
                Normals.Add(Vector3.Zero);

            foreach (var group in Geometry)
            {
                group.NIndices.Clear();
                for (int i = 0; i < group.VIndices.Count; i++)
                {
                    Int3 index = group.VIndices[i];
                    Vector3 v0 = Positions[index.X];
                    Vector3 v1 = Positions[index.Y];
                    Vector3 v2 = Positions[index.Z];

                    Vector3 leg0 = v1 - v0;
                    Vector3 leg1 = v2 - v0;
                    Vector3 normal = winding * Vector3.Cross(leg0, leg1);

                    Normals[index.X] += normal;
                    Normals[index.Y] += normal;
                    Normals[index.Z] += normal;
                    group.NIndices.Add(index);
                }
            }



            Trace.Write("..Complete!]" +Environment.NewLine);            
        }
    }


    public class OBJLoader
    {
        public string FileName { get; set; }

        public BoundingBox BBox;

        public float Scale = 1.0f;

        public bool ParseMaterials = true;
        public bool ParseNormals = true;
        public bool GenerateNormals = true;
        public bool GenerateGeometry = true;
        public bool FrongFaceWinding = true;

        public List<Vector3> Positions;
        public List<Vector3> Normals;
        public List<Vector2> Texcoords;

        protected Dictionary<string, ObjMaterial> mMtrls;
        protected string mCurrentMtrlName;
        protected ObjMaterial mCurrentMtrl;

        protected ObjGroup mCurrentGroup = null;
        protected string mCurrentGroupName;
        public List<ObjGroup> Groups;

        protected string mDirectory;

        public OBJLoader(string filename)
        {
            FileName = filename;

            if (!File.Exists(filename))
            {
                throw new FileNotFoundException("OBJLoader Error: File not found", filename);
            }

            Positions = new List<Vector3>();
            Normals = new List<Vector3>();
            Texcoords = new List<Vector2>();

            BBox = BoundingBox.Invalid;

            mMtrls = new Dictionary<string, ObjMaterial>(StringComparer.InvariantCultureIgnoreCase);
            Groups = new List<ObjGroup>();
        }

        public virtual void LoadContent(Func<string, string, Material> materialResolver = null)
        {
            LoadObj();
            CreateGeometry();
        }

        public static ObjData LoadObj(string fileName, bool generateNormals = false)
        {
            var fi = new FileInfo(fileName);
            Trace.WriteLine("Loading: " + Path.GetFileName(fileName) + "... ");
            Trace.WriteLine(string.Format("..File size {0} bytes", fi.Length));

            var obj = new OBJLoader(fileName);
            var objData = new ObjData() { FileName = fileName };
            obj.GenerateNormals = generateNormals;
            obj.LoadObj();
            objData.Geometry = obj.Groups.ToArray();
            objData.Materials = obj.mMtrls;
            objData.Bounds = obj.BBox;
            objData.Normals = obj.Normals;
            objData.Positions = obj.Positions;
            objData.Texcoords = obj.Texcoords;

            objData.CalculateBoundingSphere();
            Trace.WriteLine(string.Format("{0} Meshes, {1} triangles, {2} Vertex data size", objData.Geometry.Length, objData.Positions.Count / 3, objData.Positions.Count * 12));

            return objData;
        }

        protected float LoadObj()
        {
            Positions.Clear();
            Normals.Clear();
            Texcoords.Clear();

            Groups.Clear();
            mMtrls.Clear();

            mCurrentGroup = null;
            mCurrentMtrlName = null;
            mCurrentMtrl = default(ObjMaterial);

            float start = 0.0f;

            mDirectory = Path.GetDirectoryName(FileName) + "\\";

            mCurrentGroup = new ObjGroup("default", "default");
            Groups.Add(mCurrentGroup);

            ParseFile(ParseObjToken, FileName);
            foreach (var objGroup in Groups)
            {
                if (objGroup.TexCoords.Any())
                    Texcoords.AddRange(objGroup.TexCoords);//NormalizeTexCoords(
            }
            CreateNormals();
            return start;
        }

        public Dictionary<string, ObjMaterial> GetMaterials()
        {
            return mMtrls;
        }

        public IList<ObjGroup> GetGeometry()
        {
            return this.Groups;
        }

        protected static char[] omitchars = new char[] { ' ', '\r', '\n', '\t', '/' };
        protected void ParseFile(ParseHandler parser, string filename)
        {
            StreamReader reader = new StreamReader(filename);

            while (!reader.EndOfStream)
            {
                string line = reader.ReadLine();

                if (string.IsNullOrEmpty(line))
                    continue;

                string[] tokens = line.Split(omitchars, StringSplitOptions.RemoveEmptyEntries);

                if (tokens == null || tokens[0].StartsWith("#"))
                    continue;

                parser(line, tokens);
            }
        }

        protected void ParseObjToken(string line, string[] tokens)
        {
            string token = tokens[0].ToLower();
            int lastObjTexCoord = 0;
            try
            {
                switch (token)
                {
                    case "v":
                        {
                            float x = float.Parse(tokens[1], CultureInfo.InvariantCulture);
                            float y = float.Parse(tokens[2], CultureInfo.InvariantCulture);
                            float z = float.Parse(tokens[3], CultureInfo.InvariantCulture);

                            Vector3 p = new Vector3(x * Scale, y * Scale, z * Scale);
                            BBox.AddFloat3(p);

                            Positions.Add(p);
                            break;
                        }
                    case "vn":
                        {
                            if (ParseNormals)
                            {
                                float x = float.Parse(tokens[1], CultureInfo.InvariantCulture);
                                float y = float.Parse(tokens[2], CultureInfo.InvariantCulture);
                                float z = float.Parse(tokens[3], CultureInfo.InvariantCulture);

                                Normals.Add(new Vector3(x, y, z));
                            }
                            break;
                        }
                    case "vt":
                        {
                            float x = float.Parse(tokens[1], CultureInfo.InvariantCulture);
                            float y = float.Parse(tokens[2], CultureInfo.InvariantCulture);

                            //Texcoords.Add(new Vector2(x,y));
                            mCurrentGroup.TexCoords.Add(new Vector2(x, y));
                            break;
                        }
                    case "f":
                        {
                            int delta = 1;
                            int v0 = 0, v1 = 0, v2 = 0;

                            if (tokens.Length >= 10)
                            {
                                //full v, vn, vt indices
                                v0 = int.Parse(tokens[1]);
                                v1 = int.Parse(tokens[4]);
                                v2 = int.Parse(tokens[7]);

                                mCurrentGroup.VIndices.Add(new Int3(v0 - delta, v1 - delta, v2 - delta));

                                v0 = int.Parse(tokens[2]);
                                v1 = int.Parse(tokens[5]);
                                v2 = int.Parse(tokens[8]);

                                mCurrentGroup.TIndices.Add(new Int3(v0 - delta, v1 - delta, v2 - delta));

                                if (ParseNormals)
                                {
                                    v0 = int.Parse(tokens[3]);
                                    v1 = int.Parse(tokens[6]);
                                    v2 = int.Parse(tokens[9]);

                                    mCurrentGroup.NIndices.Add(new Int3(v0 - delta, v1 - delta, v2 - delta));
                                }

                            }
                            else if (tokens.Length >= 7)
                            {
                                v0 = int.Parse(tokens[1]);
                                v1 = int.Parse(tokens[3]);
                                v2 = int.Parse(tokens[5]);

                                mCurrentGroup.VIndices.Add(new Int3(v0 - delta, v1 - delta, v2 - delta));

                                v0 = int.Parse(tokens[2]);
                                v1 = int.Parse(tokens[4]);
                                v2 = int.Parse(tokens[6]);

                                if (line.Contains("//"))
                                {
                                    if (ParseNormals)
                                        mCurrentGroup.NIndices.Add(new Int3(v0 - delta, v1 - delta, v2 - delta));
                                }
                                else
                                {
                                    mCurrentGroup.TIndices.Add(new Int3(v0 - delta, v1 - delta, v2 - delta));
                                }
                            }
                            else if (tokens.Length >= 4)
                            {
                                //v indices
                                v0 = int.Parse(tokens[1]);
                                v1 = int.Parse(tokens[2]);
                                v2 = int.Parse(tokens[3]);

                                mCurrentGroup.VIndices.Add(new Int3(v0 - delta, v1 - delta, v2 - delta));
                            }

                            break;
                        }
                    case "g":
                        {
                            if (!ParseMaterials)
                                break;

                            string name = tokens[1];
                            mCurrentGroupName = name;
                            lastObjTexCoord = Texcoords.Count;
                            mCurrentGroup = new ObjGroup(name, name) { mtrl = mCurrentMtrlName };
                            Groups.Add(mCurrentGroup);
                            break;
                        }
                    case "usemtl":
                        {
                            if (!ParseMaterials)
                                break;

                            string name = tokens[1];
                            //if (mCurrentGroup != null){ Texcoords.AddRange(NormalizeTexCoords(mCurrentGroup.TexCoords));}
                            //string groupname = mCurrentGroupName + name;

                            //mCurrentGroup = new ObjGroup( groupname, mCurrentGroupName );
                            //Groups.Add( mCurrentGroup );

                            mCurrentMtrlName = name;
                            mCurrentGroup.mtrl = name;

                            break;
                        }
                    case "mtllib":
                        {
                            if (!ParseMaterials)
                                break;

                            mCurrentMtrl = default(ObjMaterial);

                            string name = mDirectory + tokens[1];
                            ParseFile(ParseMtrlToken, name);

                            if (mCurrentMtrlName != null)
                            {
                                mMtrls.Add(mCurrentMtrlName, mCurrentMtrl);
                            }
                            mCurrentMtrlName = null;
                            break;
                        }
                    default:
                        break;
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine("Error parsing obj token: " + ex.Message);
            }
        }

        private List<Vector2> NormalizeTexCoords(List<Vector2> texCoords)
        {
            float maxX = texCoords.Select(c => Math.Abs(c.X)).Max();
            float maxY = texCoords.Select(c => Math.Abs(c.Y)).Max();
            List<Vector2> result = new List<Vector2>();
            foreach (var texCoord in texCoords)
            {
                var tc = new Vector2(Math.Abs(texCoord.X) > 1.0 ? texCoord.X / maxX : texCoord.X,
                    Math.Abs(texCoord.Y) > 1.0 ? texCoord.Y / maxY : texCoord.Y);
                result.Add(tc);
            }
            return result;
        }

        protected void ParseMtrlToken(string line, string[] tokens)
        {
            string token = tokens[0].ToLower();
            try
            {
                switch (token)
                {
                    case "newmtl":
                        {
                            if (mCurrentMtrlName != null)
                            {
                                mMtrls.Add(mCurrentMtrlName, mCurrentMtrl);
                            }

                            mCurrentMtrlName = tokens[1];
                            mCurrentMtrl = new ObjMaterial() { Name = mCurrentMtrlName };
                            break;
                        }
                    case "ns":
                        {
                            mCurrentMtrl.Ns = float.Parse(tokens[1], CultureInfo.InvariantCulture);
                            break;
                        }
                    case "ni":
                        {
                            mCurrentMtrl.Ni = float.Parse(tokens[1], CultureInfo.InvariantCulture);
                            break;
                        }
                    case "d":
                        {
                            mCurrentMtrl.d = float.Parse(tokens[1], CultureInfo.InvariantCulture);
                            break;
                        }
                    case "tr":
                        {
                            mCurrentMtrl.Tr = float.Parse(tokens[1], CultureInfo.InvariantCulture);
                            break;
                        }
                    case "illum":
                        {
                            mCurrentMtrl.illum = float.Parse(tokens[1], CultureInfo.InvariantCulture);
                            break;
                        }
                    case "tf":
                        {
                            float x = float.Parse(tokens[1], CultureInfo.InvariantCulture);
                            float y = float.Parse(tokens[2], CultureInfo.InvariantCulture);
                            float z = float.Parse(tokens[3], CultureInfo.InvariantCulture);
                            mCurrentMtrl.Tf = new Vector3(x, y, z);
                            break;
                        }
                    case "ka":
                        {
                            float x = float.Parse(tokens[1], CultureInfo.InvariantCulture);
                            float y = float.Parse(tokens[2], CultureInfo.InvariantCulture);
                            float z = float.Parse(tokens[3], CultureInfo.InvariantCulture);
                            mCurrentMtrl.Ka = new Vector3(x, y, z);
                            break;
                        }
                    case "kd":
                        {
                            float x = float.Parse(tokens[1], CultureInfo.InvariantCulture);
                            float y = float.Parse(tokens[2], CultureInfo.InvariantCulture);
                            float z = float.Parse(tokens[3], CultureInfo.InvariantCulture);
                            mCurrentMtrl.Kd = new Vector3(x, y, z);
                            break;
                        }
                    case "ks":
                        {
                            float x = float.Parse(tokens[1], CultureInfo.InvariantCulture);
                            float y = float.Parse(tokens[2], CultureInfo.InvariantCulture);
                            float z = float.Parse(tokens[3], CultureInfo.InvariantCulture);
                            mCurrentMtrl.Ks = new Vector3(x, y, z);
                            break;
                        }
                    case "ke":
                        {
                            float x = float.Parse(tokens[1], CultureInfo.InvariantCulture);
                            float y = float.Parse(tokens[2], CultureInfo.InvariantCulture);
                            float z = float.Parse(tokens[3], CultureInfo.InvariantCulture);
                            mCurrentMtrl.Ke = new Vector3(x, y, z);
                            break;
                        }
                    case "bump":
                        {
                            if (tokens.Length == 1)
                                return;
                            mCurrentMtrl.BumpTexture = tokens.Length == 2 || tokens.Length > 3 ? tokens.Last() : line.Substring(line.IndexOf(tokens[1], StringComparison.InvariantCultureIgnoreCase));
                            break;
                        }
                    case "map_kd":
                        {
                            if (tokens.Length == 1)
                                return;
                            mCurrentMtrl.DiffuseTexture = tokens.Length == 2 ? tokens.Last() : line.Substring(line.IndexOf(tokens[1], StringComparison.InvariantCultureIgnoreCase));
                            break;
                        }
                    case "map_ks":
                        if (tokens.Length == 1)
                            return;
                        mCurrentMtrl.GlossTexture = tokens.Length == 2 ? tokens.Last() : line.Substring(line.IndexOf(tokens[1], StringComparison.InvariantCultureIgnoreCase));
                        break;
                    case "map_ka":
                        if (tokens.Length == 1)
                            return;
                        mCurrentMtrl.AmbientTexture = tokens.Length == 2 ? tokens.Last() : line.Substring(line.IndexOf(tokens[1], StringComparison.InvariantCultureIgnoreCase));
                        break;
                    case "map_ns":
                        if (tokens.Length == 1)
                            return;
                        mCurrentMtrl.FresnelTexture = tokens.Length == 2 ? tokens.Last() : line.Substring(line.IndexOf(tokens[1], StringComparison.InvariantCultureIgnoreCase));
                        break;
                    case "map_refl":
                    case "refl":
                        {
                            if (tokens.Length == 1)
                                return;
                            mCurrentMtrl.SpecularTexture = tokens.Length == 2 ? tokens.Last() : line.Substring(line.IndexOf(tokens[1], StringComparison.InvariantCultureIgnoreCase));
                            break;
                        }
                    case "map_opacity":
                    case "map_d":
                        {
                            if (tokens.Length == 1)
                                return;
                            mCurrentMtrl.AlphaTexture = tokens.Length == 2 ? tokens.Last() : line.Substring(line.IndexOf(tokens[1], StringComparison.InvariantCultureIgnoreCase));
                            break;
                        }
                    default:
                        break;
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine("Error parsing mtrl token: " + ex.Message);
            }
        }

        protected virtual void CreateGeometry()
        {
            if (GenerateGeometry == false)
                return;
        }


        protected virtual void CreateGeometry(Func<string, string, Material> materialResolver)
        {
            if (GenerateGeometry == false)
                return;
        }

        protected void CreateNormals()
        {
            if (Normals.Count > 0 || !GenerateNormals)
            {
                GenerateNormals = false;
                return;
            }

            float winding = 1.0f;
            if (FrongFaceWinding == false)
                winding = -1.0f;

            Trace.Write("[Creating normals: ");
            for (int i = 0; i < Positions.Count; i++)
                Normals.Add(Vector3.Zero);

            //loop through the triangles
            foreach (var group in Groups)
            {
                for (int i = 0; i < group.VIndices.Count; i++)
                {
                    Int3 index = group.VIndices[i];
                    Vector3 v0 = Positions[index.X];
                    Vector3 v1 = Positions[index.Y];
                    Vector3 v2 = Positions[index.Z];

                    Vector3 leg0 = v1 - v0;
                    Vector3 leg1 = v2 - v0;
                    Vector3 normal = winding * Vector3.Cross(leg0, leg1);

                    Normals[index.X] += normal;
                    Normals[index.Y] += normal;
                    Normals[index.Z] += normal;
                }
            }

        }

        protected bool ValidateGroup(ObjGroup group)
        {
            for (int i = 0; i < group.VIndices.Count; i++)
            {
                System.Diagnostics.Debug.Assert(group.VIndices[i].X < Positions.Count);
                System.Diagnostics.Debug.Assert(group.VIndices[i].Y < Positions.Count);
                System.Diagnostics.Debug.Assert(group.VIndices[i].Z < Positions.Count);

                if (i < group.NIndices.Count && group.NIndices.Count > 0 && Normals.Count > 0)
                {
                    System.Diagnostics.Debug.Assert(group.NIndices[i].X < Normals.Count);
                    System.Diagnostics.Debug.Assert(group.NIndices[i].Y < Normals.Count);
                    System.Diagnostics.Debug.Assert(group.NIndices[i].Z < Normals.Count);
                }

                if (i < group.TIndices.Count && group.TIndices.Count > 0 && Texcoords.Count > 0)
                {
                    System.Diagnostics.Debug.Assert(group.TIndices[i].X < Texcoords.Count);
                    System.Diagnostics.Debug.Assert(group.TIndices[i].Y < Texcoords.Count);
                    System.Diagnostics.Debug.Assert(group.TIndices[i].Z < Texcoords.Count);
                }
            }
            return true;
        }

        public void WriteExplodedOBJ(string path)
        {
            StreamWriter writer = new StreamWriter(path);

            int prevGroupVertCount = 0;
            int vertexCount = 0;
            foreach (ObjGroup group in Groups)
            {
                //empty group
                if (group.VIndices.Count == 0 && group.NIndices.Count == 0 && group.TIndices.Count == 0)
                    continue;

                writer.WriteLine("\n#\n# object {0}\n#", group.group);

                for (int i = 0; i < group.VIndices.Count; i++)
                {
                    Int3 index = group.VIndices[i];
                    writer.WriteLine("v {0} {1} {2}", Positions[index.X].X, Positions[index.X].Y, Positions[index.X].Z);
                    writer.WriteLine("v {0} {1} {2}", Positions[index.Y].X, Positions[index.Y].Y, Positions[index.Y].Z);
                    writer.WriteLine("v {0} {1} {2}", Positions[index.Z].X, Positions[index.Z].Y, Positions[index.Z].Z);

                    vertexCount += 3;
                }
                writer.WriteLine("# {0} vertices\n", group.VIndices.Count * 3);

                for (int i = 0; i < group.NIndices.Count; i++)
                {
                    Int3 index = group.NIndices[i];
                    writer.WriteLine("vn {0} {1} {2}", Normals[index.X].X, Normals[index.X].Y, Normals[index.X].Z);
                    writer.WriteLine("vn {0} {1} {2}", Normals[index.Y].X, Normals[index.Y].Y, Normals[index.Y].Z);
                    writer.WriteLine("vn {0} {1} {2}", Normals[index.Z].X, Normals[index.Z].Y, Normals[index.Z].Z);
                }
                writer.WriteLine("# {0} normals\n", group.NIndices.Count * 3);

                for (int i = 0; i < group.TIndices.Count; i++)
                {
                    Int3 index = group.TIndices[i];
                    writer.WriteLine("vt {0} {1}", Texcoords[index.X].X, Texcoords[index.X].Y);
                    writer.WriteLine("vt {0} {1}", Texcoords[index.Y].X, Texcoords[index.Y].Y);
                    writer.WriteLine("vt {0} {1}", Texcoords[index.Z].X, Texcoords[index.Z].Y);
                }
                writer.WriteLine("# {0} texture coordinates\n", group.TIndices.Count * 3);

                //triangles
                writer.WriteLine("g {0}", group.group);
                writer.WriteLine("usemtl {0}", group.mtrl);

                for (int i = prevGroupVertCount; i < vertexCount; i += 3)
                {
                    if (group.TIndices.Count == 0 && group.NIndices.Count > 0)
                    {
                        Int3 vIndex = group.VIndices[i];
                        Int3 nIndex = group.NIndices[i];

                        writer.WriteLine("f {0}//{1} {2}//{3} {4}//{5}", vIndex.X + 1, nIndex.X + 1,
                                                                          vIndex.Y + 1, nIndex.Y + 1,
                                                                          vIndex.Z + 1, nIndex.Z + 1);
                    }
                    else if (group.NIndices.Count == 0 && group.TIndices.Count > 0)
                    {
                        Int3 vIndex = group.VIndices[i];
                        Int3 tIndex = group.TIndices[i];

                        writer.WriteLine("f {0}/{1} {2}/{3} {4}/{5}", vIndex.X + 1, tIndex.X + 1,
                                                                       vIndex.Y + 1, tIndex.Y + 1,
                                                                       vIndex.Z + 1, tIndex.Z + 1);
                    }
                    else if (group.NIndices.Count == 0 && group.TIndices.Count == 0)
                    {
                        Int3 vIndex = group.VIndices[i];

                        writer.WriteLine("f {0} {1} {2}", vIndex.X, vIndex.Y, vIndex.Z);
                    }
                    else
                    {
                        writer.WriteLine("f {0}/{1}/{2} {3}/{4}/{5} {6}/{7}/{8}", i + 1, i + 1, i + 1,
                                                                                   i + 2, i + 2, i + 2,
                                                                                   i + 3, i + 3, i + 3);
                    }
                }

                prevGroupVertCount = vertexCount;
            }

            writer.Close();
        }
    }
}
