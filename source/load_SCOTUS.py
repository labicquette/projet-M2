# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FrenchMedMCQA : A French Multiple-Choice Question Answering Corpus for Medical domain"""

import os
import json

import datasets

_DESCRIPTION = """\
SCOTUS
"""


class SCOTUS(datasets.GeneratorBasedBuilder):
    """SCOTUS : An English Dataset on the opinions of the SCOTUS"""

    VERSION = datasets.Version("1.0.0")

    def _info(self):

        features = datasets.Features(
            {
                "justia_link" : datasets.Value("string"),
                "docket_number" : datasets.Value("string"),
                "year_argued" : datasets.Value("string"),
                "court_name" : datasets.Value("string"),
                "majority_opinion_author" : datasets.Value("string"),
                "raw_source" : datasets.Value("string"),
                "opinion_texts_source" : datasets.Sequence(
                    datasets.Value("string")
                ),
                "raw_target" : {
                    "facts_of_the_case": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "conclusion": datasets.Value("string"),
                },
                "case_name" : datasets.Value("string"),
                "description" : datasets.Value("string"),
                "first_party_label" : datasets.Value("string"),
                "second_party_label" : datasets.Value("string"),
                "decisions" : datasets.Sequence({
                    "decision_description": datasets.Value("string"),
                    "decision_majority_vote": datasets.Value("int32"),
                    "decision_minority_vote": datasets.Value("int32"),
                    "decision_winning_party": datasets.Value("string"),
                    "decision_decision_type": datasets.Value("string"),
                    }),
                "written_opinion" : datasets.Sequence({
                    "id" : datasets.Value("int32"),
                    "title" : datasets.Value("string"),
                    "judge_full_name" : datasets.Value("string"),
                    "judge_last_name" : datasets.Value("string")
                    }),
            }
        )
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join("./data/SCOTUS/", "train.json"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join("./data/SCOTUS/", "dev.json"),
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     gen_kwargs={
            #         "filepath": os.path.join("../json/", "test.json"),
            #     },
            # ),
        ]

    def _generate_examples(self, filepath):

        with open(filepath, encoding="utf-8") as f:

            data = json.load(f)

            for key, d in enumerate(data):

                yield key, {
                    "justia_link": d["justia_link"],
                    "docket_number": d["docket_number"],
                    "year_argued": d["year_argued"],
                    "court_name": d["court_name"],
                    "majority_opinion_author": d["majority_opinion_author"],
                    "raw_source": d["raw_source"],
                    "opinion_texts_source": d["opinion_texts_source"],
                    "raw_target": {k:v for k,v in d["raw_target"].items()},
                    "case_name" : d["case_name"],
                    "description": d["description"],
                    "first_party_label": d["first_party_label"],
                    "second_party_label": d["second_party_label"],
                    "decisions" : d["decisions"],
                    "written_opinion" : [{k:v for k,v in dic.items()} for dic in d["written_opinion"]],
                }
