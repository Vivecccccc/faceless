import elasticsearch as es

from ...utils.constants import ES_CONSTANTS

ENDPOINT_URL = ES_CONSTANTS['ENDPOINT_URL']
BASIC_AUTH = ES_CONSTANTS['BASIC_AUTH']
INDEX_NAME = ES_CONSTANTS['INDEX_NAME']

es_client = es.Elasticsearch([ENDPOINT_URL], basic_auth=BASIC_AUTH, ca_certs=ES_CONSTANTS['CA_CERTS'])