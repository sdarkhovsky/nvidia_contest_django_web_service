from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie, csrf_protect
from .rag_langchain_nvidia_api import create_embeddings, create_chat_summarization_qa, create_chat_qa

#@ensure_csrf_cookie
@csrf_exempt
#@csrf_protect
def process_user_message(request):

    # request.body contains the image
    # print("request.body: ", request.body)

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