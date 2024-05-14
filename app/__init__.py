import logging
import os

from flask import Flask, redirect, url_for
from flask_cors import CORS

from app.algorithm import algorithm_bp
from app.web import web_bp


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True, static_folder='static', template_folder='./templates')

    # cross domain
    cors = CORS(app, resources={r'/*': {'origins': '*'}})

    app.config.from_mapping(
        SECRET_KEY='pathway_gnn',
        # DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )
    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    """*****Log*******"""
    # 设置日志级别为DEBUG
    # app.logger.setLevel(logging.DEBUG)
    #
    # # 创建一个StreamHandler，将日志输出到控制台
    # handler = logging.StreamHandler()
    # handler.setLevel(logging.DEBUG)
    #
    # # 创建一个Formatter，用于设置日志输出的格式
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)
    #
    # # 添加StreamHandler到Flask应用程序的日志处理程序中
    # app.logger.addHandler(handler)
    """*****Log*******"""

    # register the blueprint
    app.register_blueprint(web_bp)
    app.register_blueprint(algorithm_bp)

    return app


app = create_app()


# app.run()
#
# for i in range(len(app.url_map.__dict__['_rules'])):
#     print(app.url_map.__dict__['_rules'][i])


@app.route('/')
def home_redirect():
    # print(url_for('web.homepage'))
    return redirect(url_for('web.homepage'))
    # return redirect("/web/home")
