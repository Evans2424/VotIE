#!/usr/bin/env python3
"""Create short examples with proper JSON formatting."""
import json

examples = [
  {
    "id": "Fundao_cm_002_2022-14-02_seg003",
    "text": "A Câmara Municipal, em reunião realizada por videoconferência, tomou conhecimento e deliberou, por maioria e em minuta, aprovar a proposta apresentada. (Aprovação da Estrutura Tarifária para o ano de 2022 – abastecimento de água e saneamento de águas residuais) Votaram contra, os Senhores Vereadores Dra. Joana Bento e Prof. Sérgio Mendes.",
    "municipality": "M04",
    "has_voting_event": True,
    "event": {
      "subject": "Aprovação da Estrutura Tarifária para o ano de 2022 – abastecimento de água e saneamento de águas residuais",
      "participants": [
        {"text": "A Câmara Municipal", "position": "Favor"},
        {"text": "Joana Bento", "position": "Against"},
        {"text": "Sérgio Mendes", "position": "Against"}
      ],
      "counting": [{"text": "por maioria", "type": "Majority"}],
      "voting_expressions": ["deliberou"]
    }
  },
  {
    "id": "Campomaior_cm_003_2022-02-02_seg015",
    "text": "Antes de se proceder à deliberação, o senhor vereador Paulo Pinheiro ausentou-se da sala. A CÂMARA DELIBEROU, POR TODOS OS MEMBROS PRESENTES, APROVAR A PROPOSTA referente à MINUTA DO PROTOCOLO DE COLABORAÇÃO COM OS BOMBEIROS VOLUNTÁRIOS.",
    "municipality": "M02",
    "has_voting_event": True,
    "event": {
      "subject": "PROTOCOLO DE COLABORAÇÃO COM OS BOMBEIROS",
      "participants": [
        {"text": "Paulo Pinheiro", "position": "Absent"},
        {"text": "A CÂMARA", "position": "Favor"}
      ],
      "counting": [{"text": "POR TODOS OS MEMBROS PRESENTES", "type": "Unanimity"}],
      "voting_expressions": ["DELIBEROU"]
    }
  },
  {
    "id": "Porto_cm_052_2024-02-12_seg006",
    "text": "Aprovação de reconhecimento de 5 alojamentos turísticos. Aprovada por maioria, com 1 abstenção da Senhora Vereadora da CDU e com 1 voto contra da Senhora Vereadora do BE.",
    "municipality": "M06",
    "has_voting_event": True,
    "event": {
      "subject": "reconhecimento de 5 alojamentos turísticos",
      "participants": [
        {"text": "Vereadora da CDU", "position": "Abstention"},
        {"text": "Vereadora do BE", "position": "Against"}
      ],
      "counting": [{"text": "por maioria", "type": "Majority"}],
      "voting_expressions": ["Aprovada"]
    }
  },
  {
    "id": "Alandroal_cm_006_2024-02-28_seg017",
    "text": "SUBSÍDIOS DE NASCIMENTO. O Executivo Municipal deliberou por unanimidade aprovar dois subsídios e por maioria, sem a participação da Vereadora Fernanda Romão, aprovar um subsídio.",
    "municipality": "M01",
    "has_voting_event": True,
    "event": {
      "subject": "SUBSÍDIOS DE NASCIMENTO",
      "participants": [
        {"text": "o Executivo Municipal", "position": "Favor"},
        {"text": "Fernanda Romão", "position": "Absent"}
      ],
      "counting": [
        {"text": "por unanimidade", "type": "Unanimity"},
        {"text": "por maioria", "type": "Majority"}
      ],
      "voting_expressions": ["deliberou"]
    }
  },
  {
    "id": "Guimaraes_cm_016_2024-09-30_seg008",
    "text": "OPERAÇÃO DE LOTEAMENTO – QUINTA DA ADEGANHA. DELIBERADO, POR MAIORIA, INDEFERIR O RECURSO. Votaram a favor o Presidente e os Vereadores Adelina Pinto, Paulo Silva, Nelson Felgueiras e Sofia Ferreira. Votaram contra Bruno Fernandes, Vânia Silva, Hugo Ribeiro e Emília Lemos. Ana Cotter não participou.",
    "municipality": "M05",
    "has_voting_event": True,
    "event": {
      "subject": "OPERAÇÃO DE LOTEAMENTO – QUINTA DA ADEGANHA",
      "participants": [
        {"text": "o Presidente", "position": "Favor"},
        {"text": "Adelina Pinto", "position": "Favor"},
        {"text": "Paulo Silva", "position": "Favor"},
        {"text": "Nelson Felgueiras", "position": "Favor"},
        {"text": "Sofia Ferreira", "position": "Favor"},
        {"text": "Bruno Fernandes", "position": "Against"},
        {"text": "Vânia Silva", "position": "Against"},
        {"text": "Hugo Ribeiro", "position": "Against"},
        {"text": "Emília Lemos", "position": "Against"},
        {"text": "Ana Cotter", "position": "Absent"}
      ],
      "counting": [{"text": "POR MAIORIA", "type": "Majority"}],
      "voting_expressions": ["DELIBERADO"]
    }
  },
  {
    "id": "Covilha_cm_010_2024-06-17_seg004",
    "text": "ICOVI – Infraestruturas da Covilhã. Não participou José Miguel Oliveira. A Câmara deliberou, com o voto contra de Ricardo Silva e Jorge Simões, aprovar a transferência de 176.266,35€.",
    "municipality": "M03",
    "has_voting_event": True,
    "event": {
      "subject": "transferência para ICOVI",
      "participants": [
        {"text": "José Miguel Oliveira", "position": "Absent"},
        {"text": "A Câmara", "position": "Favor"},
        {"text": "Ricardo Silva", "position": "Against"},
        {"text": "Jorge Simões", "position": "Against"}
      ],
      "counting": [],
      "voting_expressions": ["deliberou"]
    }
  },
  {
    "id": "Covilha_cm_010_2024-06-17_seg003",
    "text": "COVITUR – Turismo da Covilhã. Não participou Maria Gouveia. A Câmara deliberou, com voto contra de Ricardo Silva e Jorge Simões, aprovar a transferência financeira.",
    "municipality": "M03",
    "has_voting_event": True,
    "event": {
      "subject": "transferência para COVITUR",
      "participants": [
        {"text": "Maria Gouveia", "position": "Absent"},
        {"text": "A Câmara", "position": "Favor"},
        {"text": "Ricardo Silva", "position": "Against"},
        {"text": "Jorge Simões", "position": "Against"}
      ],
      "counting": [],
      "voting_expressions": ["deliberou"]
    }
  },
  {
    "id": "Covilha_cm_010_2024-06-17_seg002",
    "text": "Contas Consolidadas de 2023. Não participaram Maria Gouveia e José Oliveira. A Câmara deliberou, com voto contra de Ricardo Silva e Jorge Simões, aprovar as Contas de 2023.",
    "municipality": "M03",
    "has_voting_event": True,
    "event": {
      "subject": "Contas Consolidadas de 2023",
      "participants": [
        {"text": "Maria Gouveia", "position": "Absent"},
        {"text": "José Oliveira", "position": "Absent"},
        {"text": "A Câmara", "position": "Favor"},
        {"text": "Ricardo Silva", "position": "Against"},
        {"text": "Jorge Simões", "position": "Against"}
      ],
      "counting": [],
      "voting_expressions": ["deliberou"]
    }
  },
  {
    "id": "Alandroal_cm_019_2024-08-07_seg009",
    "text": "APROVAÇÃO DE ATAS. O Executivo Municipal deliberou por unanimidade aprovar a ata n.º 16 e por unanimidade, sem o Vice-Presidente, aprovar a ata n.º 17 e por unanimidade, sem a Vereadora, aprovar a ata n.º 18.",
    "municipality": "M01",
    "has_voting_event": True,
    "event": {
      "subject": "APROVAÇÃO DE ATAS",
      "participants": [
        {"text": "o Executivo Municipal", "position": "Favor"},
        {"text": "o Vice-Presidente", "position": "Absent"},
        {"text": "a Vereadora", "position": "Absent"}
      ],
      "counting": [{"text": "por unanimidade", "type": "Unanimity"}],
      "voting_expressions": ["deliberou"]
    }
  },
  {
    "id": "Alandroal_cm_003_2023-02-01_seg009",
    "text": "APROVAÇÃO DE ATA. O Executivo Municipal deliberou por unanimidade, sem a eleita pelo Nós, Cidadãos, aprovar a ata n.º 02.",
    "municipality": "M01",
    "has_voting_event": True,
    "event": {
      "subject": "ata n.º 02",
      "participants": [
        {"text": "o Executivo Municipal", "position": "Favor"},
        {"text": "a eleita pelo Nós, Cidadãos", "position": "Absent"}
      ],
      "counting": [{"text": "por unanimidade", "type": "Unanimity"}],
      "voting_expressions": ["deliberou"]
    }
  }
]

output_path = '/Users/josepedroevans/Desktop/VotIE/langextract_ner/shared/short_examples.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(examples, f, indent=2, ensure_ascii=False)

print(f"✅ Created {output_path}")
print(f"📊 10 focused examples")
print(f"📝 Total characters: {sum(len(ex['text']) for ex in examples)}")
print(f"📏 Average length: {sum(len(ex['text']) for ex in examples) / len(examples):.0f} chars")
