from django.http import HttpResponse
import datetime


def process_user_message(request):
    now = datetime.datetime.now()
    html = "<html><body>It is now %s.</body></html>" % now

    query = request.GET['query']
    print("query", query)

    return HttpResponse("html")