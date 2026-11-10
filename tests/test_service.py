"""Run with unittest or pytest. Endpoint fixtures are not model-quality measurements."""
from pathlib import Path
import io,json,unittest,os,tempfile
from unittest.mock import patch
from PIL import Image
from fastapi.testclient import TestClient
from healthpal_ai.app import create_app
from healthpal_ai.policy import Policy,decide
from healthpal_ai.service import ROOT,RecognitionService,YoloDetector


def candidate(class_id=87,confidence=.8,box=None):
    return {'class_id':class_id,'confidence':confidence,'xyxy':box or [0.,0.,64.,48.]}


def image_bytes(format='PNG',exif=None):
    stream=io.BytesIO();kw={'exif':exif} if exif else {}
    Image.new('RGB',(64,48),(120,90,60)).save(stream,format=format,**kw)
    return stream.getvalue()


class Detector:
    def __init__(self,results):self.results=results;self.calls=0;self.sizes=[]
    def predict(self,image):
        value=self.results[self.calls];self.calls+=1;self.sizes.append(image.size);return value


def fixture(results):
    detector=Detector(results)
    service=RecognitionService(detector_factory=lambda *_:detector)
    return service,detector,create_app(lambda:service)


class ApiTests(unittest.TestCase):
    def test_accepted_preserves_metadata_and_requires_confirmation(self):
        service,detector,app=fixture([candidate()])
        with TestClient(app) as c:
            result=c.post('/v1/recognize',files={'file':('meal.png',image_bytes(),'image/png')},headers={'X-Request-ID':'backend-test-123'})
            self.assertEqual(result.status_code,200)
            body=result.json();self.assertEqual(body['status'],'accepted');self.assertEqual(body['policy'],'P1')
            self.assertEqual(body['request_id'],'backend-test-123');self.assertEqual(result.headers['X-Request-ID'],'backend-test-123')
            self.assertTrue(body['user_confirmation_required']);self.assertEqual(body['suggestion']['label'],'Pho')
            self.assertEqual(body['candidate']['bbox_xyxy'],[0.,0.,64.,48.]);self.assertEqual(body['brightened_inference_passes'],0)
            self.assertEqual(detector.calls,1)

    def test_uncertain_has_no_accepted_suggestion(self):
        _,detector,app=fixture([candidate(confidence=.2)])
        with TestClient(app) as c:
            body=c.post('/v1/recognize',files={'file':('meal.png',image_bytes(),'image/png')}).json()
        self.assertEqual(body['status'],'uncertain');self.assertEqual(body['reason'],'below_threshold');self.assertIsNone(body['suggestion'])
        self.assertIsNotNone(body['candidate']);self.assertEqual(detector.calls,1)

    def test_outside_catalogue_is_uncertain_even_at_high_confidence(self):
        _,_,app=fixture([candidate(class_id=91,confidence=.99)])
        with TestClient(app) as c:body=c.post('/v1/recognize',files={'file':('meal.png',image_bytes(),'image/png')}).json()
        self.assertEqual(body['reason'],'outside_catalogue');self.assertIsNone(body['suggestion'])

    def test_no_detection(self):
        _,_,app=fixture([None])
        with TestClient(app) as c:body=c.post('/v1/recognize',files={'file':('meal.png',image_bytes(),'image/png')}).json()
        self.assertEqual(body['status'],'no_detection');self.assertIsNone(body['candidate']);self.assertIsNone(body['suggestion'])

    def test_compare_reuses_one_original_and_one_bright_pass(self):
        _,detector,app=fixture([candidate(),candidate(class_id=88)])
        with TestClient(app) as c:response=c.post('/v1/compare',files={'file':('meal.png',image_bytes(),'image/png')})
        self.assertEqual(response.status_code,200);body=response.json()
        self.assertEqual([r['status'] for r in body['results']],['accepted','accepted','uncertain'])
        self.assertEqual(body['results'][2]['reason'],'view_label_disagreement');self.assertEqual(detector.calls,2)

    def test_p2_skips_bright_when_p1_rejects(self):
        _,detector,app=fixture([candidate(confidence=.2)])
        with TestClient(app) as c:body=c.post('/v1/recognize',data={'policy':'P2'},files={'file':('meal.png',image_bytes(),'image/png')}).json()
        self.assertEqual(body['status'],'uncertain');self.assertEqual(body['brightened_inference_passes'],0);self.assertEqual(detector.calls,1)

    def test_bad_inputs_have_structured_errors(self):
        _,detector,app=fixture([])
        with TestClient(app) as c:
            for raw in [b'',b'not an image']:
                response=c.post('/v1/recognize',files={'file':('meal.png',raw,'image/png')})
                self.assertEqual(response.status_code,400);self.assertIn('request_id',response.json());self.assertIn('error',response.json())
            response=c.post('/v1/recognize',data={'policy':'P3'},files={'file':('meal.png',image_bytes(),'image/png')})
            self.assertEqual(response.status_code,422);self.assertEqual(response.json()['error']['code'],'invalid_request')
            response=c.post('/v1/recognize',files={'file':('meal.gif',image_bytes('GIF'),'image/gif')})
            self.assertEqual(response.status_code,415)
            response=c.post('/v1/recognize')
            self.assertEqual(response.status_code,422)
        self.assertEqual(detector.calls,0)

    def test_byte_limit(self):
        service,detector,app=fixture([]);service.max_bytes=10
        with TestClient(app) as c:response=c.post('/v1/recognize',files={'file':('meal.png',image_bytes(),'image/png')})
        self.assertEqual(response.status_code,413);self.assertEqual(detector.calls,0)

    def test_exif_orientation_and_output_coordinates(self):
        _,detector,app=fixture([candidate()]);exif=Image.Exif();exif[274]=6
        with TestClient(app) as c:body=c.post('/v1/recognize',files={'file':('meal.jpg',image_bytes('JPEG',exif),'image/jpeg')}).json()
        self.assertEqual(detector.sizes,[(48,64)]);self.assertEqual((body['image']['width'],body['image']['height']),(48,64))

    def test_start_failure_is_unready_not_no_detection(self):
        def broken(*_):raise RuntimeError('fixture startup failure')
        service=RecognitionService(detector_factory=broken)
        with self.assertLogs('healthpal_ai',level='ERROR'):
            with TestClient(create_app(lambda:service)) as c:
                self.assertEqual(c.get('/health/live').status_code,200)
                self.assertEqual(c.get('/health/ready').status_code,503)
                response=c.post('/v1/recognize',files={'file':('meal.png',image_bytes(),'image/png')})
        self.assertEqual(response.status_code,503);self.assertEqual(response.json()['error']['code'],'model_not_ready')

    def test_ready_and_openapi(self):
        _,_,app=fixture([])
        with TestClient(app) as c:
            self.assertEqual(c.get('/health/ready').status_code,200)
            doc=c.get('/openapi.json').json()
        self.assertIn('/v1/recognize',doc['paths']);self.assertIn('/v1/compare',doc['paths'])
        self.assertEqual(doc['components']['schemas']['Policy']['enum'],['P0','P1','P2'])

    def test_framework_errors_keep_the_contract(self):
        _,_,app=fixture([])
        with TestClient(app) as c:
            for response in [c.get('/v1/recognize'),c.get('/missing-route'),
                c.post('/v1/recognize',content=b'broken multipart',headers={'Content-Type':'multipart/form-data'})]:
                self.assertIn(response.status_code,[400,404,405]);self.assertIn('error',response.json());self.assertIn('request_id',response.json())

    def test_internal_error_does_not_become_no_detection(self):
        class Broken:
            def predict(self,image):raise RuntimeError('fixture private failure detail')
        service=RecognitionService(detector_factory=lambda *_:Broken())
        with TestClient(create_app(lambda:service)) as c:
            with self.assertLogs('healthpal_ai',level='ERROR'):
                response=c.post('/v1/recognize',files={'file':('meal.png',image_bytes(),'image/png')})
        self.assertEqual(response.status_code,500);self.assertEqual(response.json()['error']['code'],'internal_error')
        self.assertNotIn('private failure',response.text)

    def test_wrong_checkpoint_fails_before_loading(self):
        service=RecognitionService()
        with tempfile.TemporaryDirectory() as temp:
            bad=Path(temp)/'wrong.pt';bad.write_bytes(b'not the selected checkpoint')
            with patch.dict(os.environ,{'HEALTHPAL_MODEL_PATH':str(bad)}):
                with self.assertRaisesRegex(RuntimeError,'checksum mismatch'):YoloDetector(service.manifest,service.config)


class FrozenPolicyTests(unittest.TestCase):
    def test_600_saved_decisions_match_final_benchmark(self):
        config=json.loads((ROOT/'config/policy-v1.json').read_text(encoding='utf-8'))
        fixture=json.loads((ROOT/'tests/fixtures/frozen-policy-cases.json').read_text(encoding='utf-8'))
        self.assertEqual(len(fixture['cases']),200)
        for case in fixture['cases']:
            for policy in Policy:
                with self.subTest(case=case['case_id'],policy=policy):
                    got=decide(case['original'],case['brightened'],policy,config)
                    self.assertEqual({'status':got.status,'reason':got.reason},case['expected'][policy])


if __name__=='__main__':unittest.main()
