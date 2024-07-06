from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from .rag_langchain_nvidia_api import create_embeddings, create_chat_summarization_qa, create_chat_qa
import datetime

#@ensure_csrf_cookie
#@csrf_exempt
def process_user_message(request):

    now = datetime.datetime.now()
    html = "<html><body>It is now %s.</body></html>" % now

    query = request.GET['query']
    print("query: ", query)

    qa = create_chat_qa()
    result = qa({"question": query})
    answer = result.get("answer")
    print("answer: ", answer)

    response = HttpResponse(answer)
    response.headers['Access-Control-Allow-Origin'] = "http://localhost:3000"
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response