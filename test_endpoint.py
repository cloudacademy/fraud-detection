import fraud_detection_endpoint
import unittest
import tempfile
import json


class EndpointTest(unittest.TestCase):
    def setUp(self):
        self.db_fd, fraud_detection_endpoint.app.config['DATABASE'] = tempfile.mkstemp()
        fraud_detection_endpoint.app.testing = True
        self.app = fraud_detection_endpoint.app.test_client()

    def tearDown(self):
        pass

    def test_no_features(self):
        response = self.app.post(u"/fraud_score", data=json.dumps(dict()), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        response_object = json.loads(response.data)
        self.assertTrue(u"error" in response_object)

        response = self.app.post(u"/fraud_score", data=json.dumps(dict(features=None)), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        response_object = json.loads(response.data)
        self.assertTrue(u"error" in response_object)

    def test_single_data_point(self):
        response = self.app.post(
            u"/fraud_score",
            data=json.dumps(dict(
                features=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )), content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response_object = json.loads(response.data)
        self.assertTrue(u"scores" in response_object)
        self.assertEqual(len(response_object[u"scores"]), 1)
        self.assertTrue(isinstance(response_object[u"scores"][0], float))

    def test_multiple_data_points(self):
        response = self.app.post(
            u"/fraud_score",
            data=json.dumps(dict(
                features=[
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ]
            )), content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response_object = json.loads(response.data)
        self.assertTrue(u"scores" in response_object)
        self.assertEqual(len(response_object[u"scores"]), 2)
        self.assertTrue(isinstance(response_object[u"scores"][0], float))
        self.assertTrue(isinstance(response_object[u"scores"][1], float))


if __name__ == '__main__':
    unittest.main()
