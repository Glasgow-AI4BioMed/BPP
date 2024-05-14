import functools

from flask import url_for, render_template
from app.bean.bean_collection import ToplevelPathway
from app.bean.dataset_factory import ToplevelPathwayFactory
from app.web import web_bp
from app.web.service.toplevel_pathway_service import toplevel_pathway_service_obj


@web_bp.route('/home', methods=('GET', 'POST'))
def homepage():

    all_toplevel_pathways: list[ToplevelPathway] = toplevel_pathway_service_obj.get_all_toplevel_pathways()

    print(url_for('static', filename='data'))

    return render_template('home.html', all_toplevel_pathways=all_toplevel_pathways)


