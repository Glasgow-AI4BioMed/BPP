from flask import (
    Blueprint
)

web_bp = Blueprint('web', __name__, url_prefix='/web')

from app.web.view.reactions_list import *
from app.web.view.home import *

