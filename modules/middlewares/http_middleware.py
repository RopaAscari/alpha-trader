import json
import sys
import time
from django.http import JsonResponse
from ..utils import snake_dict_to_camel


class RequestMiddleware:
    def __init__(self, get_response,):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        if isinstance(response, JsonResponse):
            content = response.content.decode('utf-8')
            data = json.loads(content)
            camel_data = snake_dict_to_camel(data)
            response.content = json.dumps(camel_data).encode('utf-8')

        return response

    def process_exception(self, request, exception):
        if exception:
            error = {
                'message': str(exception),
                'timestamp': time.ctime(),
            }

        return JsonResponse({"error": error}, status=500)
