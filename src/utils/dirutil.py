import os


def get_root_path() -> str:
    """
    프로젝트의 ROOT PATH를 반환하는 함수
    :return: Project ROOT PATH
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
