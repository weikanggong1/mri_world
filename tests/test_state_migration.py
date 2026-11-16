"""Software metadata fixtures; no scientific checkpoint conversion is run."""
import ast
import collections
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from brain_mri_world_model.migration_cursor import window_plan, flatten_physical_batches
from brain_mri_world_model.migration_contract import bind_contract
from brain_mri_world_model import render_migrated_training as renderer
from brain_mri_world_model import reshard_optimizer as conversion


def fixture_metadata():
    """Invented configuration for API tests, with no research-data identities."""
    return dict(world_size=6, optimizer_step=10, epoch=1, next_batch=4,
        train_examples_seen=24,
        contract=dict(epochs=50, micro_batch=1, gradient_accumulation=4,
            learning_rate=1e-5, weight_decay=1e-3, warmup_fraction=.03,
            max_grad_norm=1., sigma_distribution='logit_normal', sigma_shift=1.,
            epoch_axis_policy='one_per_image', seed=7, validation_seed=11,
            gradient_checkpointing=True, mode='full_dit', optimizer='AdamW(0.9,0.999,1e-8)',
            context_sha256='a'*64, manifest_sha256='b'*64, training_code_sha256='c'*64,
            software=dict(torch='2.6.0+cu124', diffusers='0.35.2'),
            model=dict(config_sha256='d'*64, index_sha256='e'*64, parameter_dtype='float32',
                tf32=True, attention='diffusers.WanAttnProcessor/PyTorch_SDPA'),
            data_loading=dict(workers_per_rank=8, prefetch_factor=2),
            input_index=dict(runtime_eligible_summary=dict(eligible_source_images=2),
                mandatory_runtime_eligibility=dict(version='software_fixture',subjects_sha256='f'*64,
                    source_records_sha256='0'*64,subject_count=1,canonical_source_holds=1))))


class CursorTests(unittest.TestCase):
    def logical(self, count):
        return [[(i*6+j, 1.) for j in range(6)] for i in range(count)]

    def plan(self, batches, next_batch=0):
        return list(window_plan(batches, source_world_size=6, target_world_size=8,
            micro_batch=1, gradient_accumulation=4, next_batch=next_batch))

    def test_24_examples_become_three_eight_rank_waves(self):
        window=self.plan(self.logical(4))[0]
        self.assertEqual((len(window['physical_batches']), window['valid_count'], window['padding_added']), (3,24,0))

    def test_same_valid_multiset_and_denominator_for_every_tail(self):
        for tail in range(1,5):
            for valid_last in range(1,7):
                old=self.logical(tail)
                old[-1]=[(index, float(i<valid_last)) for i,(index,_) in enumerate(old[-1])]
                window=self.plan(old)[0]
                expected=collections.Counter(i for batch in old for i,w in batch if w)
                actual=collections.Counter(i for batch in window['physical_batches'] for i,w in batch if w)
                self.assertEqual(expected,actual)
                self.assertEqual(window['valid_count'],sum(expected.values()))
                # Arbitrary scalar software gradients check original normalization.
                denominator=window['valid_count']
                original=sum(sum(6/denominator*w*(i+.125) for i,w in batch)/6 for batch in old)
                migrated=sum(sum(8/denominator*w*(i+.125) for i,w in batch)/8 for batch in window['physical_batches'])
                self.assertAlmostEqual(original,migrated,places=12)

    def test_resume_does_not_replay_logical_work(self):
        plan=self.plan(self.logical(11),4)
        self.assertEqual((plan[0]['logical_start'],plan[-1]['logical_stop']),(4,11))
        self.assertEqual(sorted(i for batch in flatten_physical_batches(plan) for i,w in batch if w),list(range(24,66)))

    def test_uncommitted_window_refused_and_epoch_end_empty(self):
        with self.assertRaisesRegex(ValueError,'BOUNDARY'):
            self.plan(self.logical(8),3)
        self.assertEqual(self.plan(self.logical(5),5),[])


class ContractTests(unittest.TestCase):
    def case(self, change=None):
        temp=tempfile.TemporaryDirectory();root=Path(temp.name);origin=root/'checkpoint-000000010';origin.mkdir()
        metadata=fixture_metadata();(origin/'metadata.json').write_text(json.dumps(metadata))
        contract=copy.deepcopy(metadata['contract'])
        attestation=root/'attestation.json'
        attestation.write_text(json.dumps(dict(state='SOURCE_TARGET_SEMANTIC_INPUT_ORDER_IDENTICAL',
            source_manifest_sha256=contract['manifest_sha256'],target_manifest_sha256=contract['manifest_sha256'],semantic_rows_sha256='1'*64)))
        if change:change(contract)
        return temp,origin,contract,attestation,metadata

    def test_original_configuration_and_cursor_authority_retained(self):
        temp,origin,contract,attestation,metadata=self.case()
        with temp:
            result=bind_contract(contract,origin,attestation,8)
            self.assertEqual(result['migration']['origin_contract'],metadata['contract'])
            self.assertEqual((result['migration']['logical_world_size'],result['migration']['physical_world_size']),(6,8))

    def test_changed_learning_rate_and_same_size_different_holds_refused(self):
        changes=[lambda c:c.update(learning_rate=1e-4),
            lambda c:c['input_index']['mandatory_runtime_eligibility'].update(subjects_sha256='2'*64)]
        for change in changes:
            temp,origin,contract,attestation,_=self.case(change)
            with temp,self.assertRaises(ValueError):bind_contract(contract,origin,attestation,8)

    def test_six_original_rngs_and_two_explicit_new_rngs(self):
        result=conversion.rng_mapping(fixture_metadata())
        self.assertEqual(result[:6],[dict(physical_rank=i,kind='original_rank_rng',source_rank=i) for i in range(6)])
        self.assertNotEqual(result[6]['seed'],result[7]['seed'])
        self.assertEqual(result,conversion.rng_mapping(fixture_metadata()))

    def test_copy_preserves_source_and_uses_independent_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source_fixture';target=Path(directory)/'target_fixture'
            source.write_bytes(b'software-copy-fixture'*1000);before=conversion.fp(source)
            result=conversion.preserve_model_copy(source,target,hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(before,conversion.fp(source));self.assertNotEqual(conversion.fp(source)[:2],conversion.fp(target)[:2])
            self.assertFalse(result['separate_postwrite_rehash'])

    def test_default_converter_does_not_import_torch_or_access_checkpoint(self):
        code="import sys;from brain_mri_world_model import reshard_optimizer;assert 'torch' not in sys.modules;sys.argv=['reshard_optimizer'];reshard_optimizer.main()"
        result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['state'],'SOURCE_ONLY_NO_IO_NO_GPU')
        tree=ast.parse(Path(conversion.__file__).read_text())
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in('step','forward','backward') for n in ast.walk(tree)))

    def test_renderer_is_default_inert_and_preserves_original_math(self):
        result=subprocess.run([sys.executable,'-B','-m','brain_mri_world_model.render_migrated_training'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['state'],'SOURCE_ONLY_NO_RENDER_NO_GPU')
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'brain_mri_world_model'
            review=renderer.render(Path(renderer.__file__).parent,output)
            self.assertTrue(all(review['unchanged_function_AST'].values()))
            self.assertTrue(all(review['unchanged_loop_math_AST'].values()))
            compile((output/'train.py').read_text(),str(output/'train.py'),'exec')
            with self.assertRaises(FileExistsError):renderer.render(Path(renderer.__file__).parent,output)

if __name__=='__main__':unittest.main()
