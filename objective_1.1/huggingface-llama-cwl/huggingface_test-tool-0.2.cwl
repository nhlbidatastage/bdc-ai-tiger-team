{
    "class": "CommandLineTool",
    "cwlVersion": "v1.2",
    "$namespaces": {
        "sbg": "https://sevenbridges.com"
    },
    "baseCommand": [
        "nvidia-smi",
        "&&",
        "python3",
        "/run_model.py"
    ],
    "inputs": [
        {
            "id": "token",
            "type": "string",
            "inputBinding": {
                "prefix": "--token",
                "shellQuote": true,
                "position": 0
            }
        }
    ],
    "outputs": [
        {
            "id": "log",
            "type": "File?",
            "outputBinding": {
                "glob": "*.txt"
            }
        }
    ],
    "label": "Huggingface test-tool-0.2",
    "requirements": [
        {
            "class": "DockerRequirement",
            "dockerPull": "images.sb.biodatacatalyst.nhlbi.nih.gov/satusky/llama-testing:huggingface-pytorch-gpu-0.2"
        }
    ],
    "hints": [
        {
            "class": "sbg:SaveLogs",
            "value": "standard.out"
        },
        {
            "class": "sbg:SaveLogs",
            "value": "standard.err"
        }
    ],
    "stdout": "standard.out",
    "stderr": "standard.err",
    "sbg:projectName": "deep learning test data",
    "sbg:revisionsInfo": [
        {
            "sbg:revision": 0,
            "sbg:modifiedBy": "satusky",
            "sbg:modifiedOn": 1730490018,
            "sbg:revisionNotes": "Copy of satusky/deep-learning-test-data/huggingface-test-tool/9"
        },
        {
            "sbg:revision": 1,
            "sbg:modifiedBy": "satusky",
            "sbg:modifiedOn": 1730490713,
            "sbg:revisionNotes": ""
        }
    ],
    "sbg:image_url": null,
    "sbg:appVersion": [
        "v1.2"
    ],
    "id": "https://api.sb.biodatacatalyst.nhlbi.nih.gov/v2/apps/satusky/deep-learning-test-data/huggingface-test-tool-0_2/1/raw/",
    "sbg:id": "satusky/deep-learning-test-data/huggingface-test-tool-0_2/1",
    "sbg:revision": 1,
    "sbg:revisionNotes": "",
    "sbg:modifiedOn": 1730490713,
    "sbg:modifiedBy": "satusky",
    "sbg:createdOn": 1730490018,
    "sbg:createdBy": "satusky",
    "sbg:project": "satusky/deep-learning-test-data",
    "sbg:sbgMaintained": false,
    "sbg:validationErrors": [],
    "sbg:contributors": [
        "satusky"
    ],
    "sbg:latestRevision": 1,
    "sbg:publisher": "sbg",
    "sbg:content_hash": "aa5f7829c22b965137ca2ba734065bd64c2cd4a6ab7d8da260267e618720f12d4",
    "sbg:workflowLanguage": "CWL"
}