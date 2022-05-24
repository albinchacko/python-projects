import requests

api_key = '42a4667872a563e0a71005c1d93c1e03'
base_url = 'http://api.openweathermap.org/data/2.5/weather'

city = input('Enter the city name: ')
request_url = f"{base_url}?q={city}&appid={api_key}&units=metric"
response = requests.get(request_url)

if response.status_code == 200:
    data = response.json()
    # print(data)
    print(f"{city.capitalize()}'s current weather")
    condition = data['weather'][0]['description']
    temperature = round(data['main']['temp'], 1)
    feels_like = round(data['main']['feels_like'], 1)
    press = data['main']['pressure']
    hum = data['main']['humidity']
    wind_speed = data['wind']['speed']

    print('Weather condition:', condition)
    print('Temperature:', f'{temperature}°C')
    print('Real feel temperature:', f'{feels_like}°C')
    print('Pressure:', f'{press}hPA')
    print('Humidity:', f'{hum}%')
    print('Wind speed:', f'{wind_speed}m/sec')

else:
    print('An error occurred')