import json
import datasets
from typing import Any, Dict, List


_DESCRIPTION = "FinLawLLM dataset"

_CITATION = """\
@misc{FinLawLLM,
  author = {Hao Shen},
  title = {FinLawLLM: 金融法律跨域大语言模型},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/FduFinTech/FinLawLLM}},
}
"""

_HOMEPAGE = "https://github.com/FduFinTech/FinLawLLM"
_LICENSE = "mit"
_URL = "finlaw_dataset.jsonl"


class ExampleDataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "conversation_id": datasets.Value("string"),
            "category": datasets.Value("string"),
            "conversations": [{"from": datasets.Value("string"), "value": datasets.Value("string")}],
            "system": datasets.Value("string"),
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        file_path = dl_manager.download(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": file_path
                }
            )
        ]

    def _generate_examples(self, filepath: str) -> Dict[int, Dict[str, Any]]:
        with open(filepath, "r", encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                conversations = []

                for conv in data["conversation"]:
                    if "human" in conv:
                        conversations.append({"from": "human", "value": conv["human"].strip()})
                    if "assistant" in conv:
                        conversations.append({"from": "gpt", "value": conv["assistant"].strip()})

                yield key, {
                    "conversation_id": data.get("conversation_id", key),
                    "category": data.get("category", ""),
                    "conversations": conversations,
                    "system": data.get("system", "")
                }   
