using System.Collections.Generic;

namespace Playground
{

    public abstract class Message
    {
    }

    public class Message1 : Message
    {
        public int Data { get; set; }
    }

    public class Message2 : Message
    {
        public string Data { get; set; }
    }

    /* */

    public class JsonParser
    {
        public static string json = @"{
        ""type"": ""message-1"",
        ""data"": 42
    },

    {
    ""type"": ""message-2"",
    ""data"": ""42""
}";


        public static IEnumerable<Message> GetMessages(string js = null)
        {
            if (js == null)
            {
                js = json;
            }
            var jobject = JObject.Parse(json);

            var result = new List<Message>();

            return result;
        }
    }

}