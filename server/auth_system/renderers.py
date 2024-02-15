from rest_framework.renderers import JSONRenderer
import json

class UserJsonRenderer(JSONRenderer):
    charset = 'utf-8'

    def render(self, data, accepted_media_type=None, renderer_context=None):

        errors = data.get('errors', None)

        if errors is not None:
            return super(UserJsonRenderer,self).render(data)

        token = data.get('token')

        if token is not None and isinstance(token, bytes):
            data['token'] = token.decode('utf-8')

        return json.dumps({
            'user': data
        })