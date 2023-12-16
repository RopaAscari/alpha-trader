import requests


class HttpClient:
    """
    A generic HTTP client class that provides methods for making HTTP requests.
    """

    def __init__(self, base_url):
        self.base_url = base_url

    def get(self, url='', params=None, headers=None):
        """
        Sends a GET request to the specified URL and returns the response.
        """
        url = self.base_url + url
        response = requests.get(url, params=params, headers=headers)
        return response

    def post(self, url='', data=None, json=None, headers=None):
        """
        Sends a POST request to the specified URL and returns the response.
        """
        url = self.base_url + url
        response = requests.post(url, data=data, json=json, headers=headers)
        return response

    def put(self, url='', data=None, headers=None):
        """
        Sends a PUT request to the specified URL and returns the response.
        """
        url = self.base_url + url
        response = requests.put(url, data=data, headers=headers)
        return response

    def delete(self, url='', headers=None):
        """
        Sends a DELETE request to the specified URL and returns the response.
        """
        url = self.base_url + url
        response = requests.delete(url, headers=headers)
        return response
