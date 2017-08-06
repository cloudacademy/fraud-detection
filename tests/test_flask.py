""" Flask app tests """
from __future__ import unicode_literals
import sys
import os
import unittest
import tempfile
import json
from flask import url_for

sys.path.insert(0, os.path.join('..', os.path.dirname(__file__), 'src'))

from flask_app import app


class EndpointTest(unittest.TestCase):

    def setUp(self):
        app.config['DATABASE'] = tempfile.mkstemp()
        app.testing = True
        self.client = app.test_client()
        self.endpoint = "/predict"

    def tearDown(self):
        pass

    def post(self, data):
        return self.client.post(
            path=self.endpoint,
            data=json.dumps(data),
            content_type='application/json',
        )

    def test_empty_body(self):
        with app.app_context():
            response = self.post({})
            self.assertEqual(response.status_code, 400)
            response_object = json.loads(response.data)
            self.assertTrue("error" in response_object)

    def test_no_features(self):
        with app.app_context():
            response = self.post(dict(features=None))
            self.assertEqual(response.status_code, 400)
            response_object = json.loads(response.data)
            self.assertTrue("error" in response_object)

    def test_single_data_point(self):
        features = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]
        with app.app_context():
            response = self.post(dict(features=features))
            self.assertEqual(response.status_code, 200)
            response_object = json.loads(response.data)
            self.assertTrue("scores" in response_object)
            self.assertEqual(len(response_object["scores"]), 1)
            self.assertTrue(isinstance(response_object["scores"][0], float))

    def test_multiple_data_points(self):
        features = [
            [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        ]
        with app.app_context():
            response = self.post(dict(features=features))
            self.assertEqual(response.status_code, 200)
            response_object = json.loads(response.data)
            self.assertTrue("scores" in response_object)
            self.assertEqual(len(response_object["scores"]), 2)
            self.assertTrue(isinstance(response_object["scores"][0], float))
            self.assertTrue(isinstance(response_object["scores"][1], float))


if __name__ == '__main__':
    unittest.main()
