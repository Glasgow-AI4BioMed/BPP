from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

algorithm_bp = Blueprint('algorithm', __name__, url_prefix='/algorithm')\


from app.algorithm.view.model_process import *


