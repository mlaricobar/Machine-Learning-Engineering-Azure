import requests
import json

# URL for the web service, should be similar to:
scoring_uri = 'http://d25816c7-213d-4011-a940-892d59545e0f.eastus2.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = 'w7Nuinf2vMFxbThjOt8Wci6HCgOuQanG'

# Two sets of data to score, so we get two results back
data = {"data":
        [
        {'key_value': 62,
        'edad': -1.3623451,
        'sexo': 1.0,
        'est_cvl': 4.0,
        'sit_lab': 1.0,
        'ctd_hijos': 0.0,
        'flg_sin_email': 1.0,
        'ctd_veh': 0.0,
        'tip_lvledu': 7.0,
        'total_mean_of_saldo_count_per_month': 2.5,
        'total_sum_of_saldo_count_per_month': 30,
        'total_min_of_saldo_count_per_month': 2,
        'total_max_of_saldo_count_per_month': 4,
        'total_mean_of_saldo_sum_per_month': 2.3570971,
        'total_sum_of_saldo_sum_per_month': 28.285166,
        'total_min_of_saldo_sum_per_month': 1.8607771,
        'total_max_of_saldo_sum_per_month': 3.949139,
        'total_min_of_saldo_min_per_month': 0.92711896,
        'total_mean_of_saldo_min_per_month': 0.9273224,
        'total_max_of_saldo_max_per_month': 1.04637,
        'total_mean_of_saldo_max_per_month': 0.9552064,
        'total_min_of_condicion_min_per_month': 0,
        'total_mean_of_condicion_min_per_month': 0.0,
        'total_max_of_condicion_max_per_month': 0,
        'total_mean_of_condicion_max_per_month': 0.0,
        'total_mean_of_condicion_mean_per_month': 0.0
        },
        {'key_value': 21,
            'edad': 0.15243018,
            'sexo': 0.0,
            'est_cvl': 4.0,
            'sit_lab': 1.0,
            'ctd_hijos': 0.0,
            'flg_sin_email': 1.0,
            'ctd_veh': 0.0,
            'tip_lvledu': 0.0,
            'total_mean_of_saldo_count_per_month': 13.416666666666666,
            'total_sum_of_saldo_count_per_month': 161,
            'total_min_of_saldo_count_per_month': 6,
            'total_max_of_saldo_count_per_month': 20,
            'total_mean_of_saldo_sum_per_month': 13.418651,
            'total_sum_of_saldo_sum_per_month': 161.0238,
            'total_min_of_saldo_sum_per_month': 6.3681345,
            'total_max_of_saldo_sum_per_month': 19.67877,
            'total_min_of_saldo_min_per_month': 0.92711896,
            'total_mean_of_saldo_min_per_month': 0.92711896,
            'total_max_of_saldo_max_per_month': 1.1543791,
            'total_mean_of_saldo_max_per_month': 1.1356059,
            'total_min_of_condicion_min_per_month': 0,
            'total_mean_of_condicion_min_per_month': 0.0,
            'total_max_of_condicion_max_per_month': 333,
            'total_mean_of_condicion_max_per_month': 171.08333333333334,
            'total_mean_of_condicion_mean_per_month': 52.30934095860567
        }
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
print("############################")
print("Expected result: [0, 1]")