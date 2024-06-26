def time_to_seconds(time: str) -> float:
    return int(time.split(':')[0]) * 3600 + int(time.split(':')[1]) * 60 + float(time.split(':')[2])

def seconds_to_time(seconds: float) -> str:
    return f'{int(seconds // 3600)}:{int(seconds % 3600 // 60)}:{seconds % 60}'