from datetime import datetime

FORMAT = "%Y-%m-%d-%H-%M"


def get_now_stamp() -> str:
    return datetime.now().strftime(FORMAT)


def from_timestamp(timestamp: str) -> datetime:
    return datetime.strptime(timestamp, FORMAT)
