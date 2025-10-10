"""
Portuguese Voting Prompts for Gemini 2.0 Flash

Few-shot examples and prompts for voting entity extraction.
"""

from typing import List, Dict, Any
import langextract as lx
import logging
import warnings

# Suppress alignment warnings
warnings.filterwarnings("ignore", message=".*Prompt alignment: non-exact match.*")
logging.getLogger('absl').setLevel(logging.ERROR)


# Portuguese extraction prompt
PROMPT = """
Extraia todas as entidades relacionadas com votação em documentos de câmaras municipais portuguesas.

Tipos de entidades a identificar:
- VOTER-FAVOR: Pessoa ou entidade que votou a favor
- VOTER-AGAINST: Pessoa ou entidade que votou contra
- VOTER-ABSTENTION: Pessoa ou entidade que se absteve
- VOTER-ABSENT: Pessoa ou entidade ausente da votação
- SUBJECT: Assunto ou matéria sendo votada (apenas substantivos/frases nominais, SEM verbos de ação como "aprovar", "ratificar", "deliberar")
- VOTING: Ação de votar (deliberou, aprovou, rejeitou, etc.)
- COUNTING-MAJORITY: Contabilização por maioria
- COUNTING-UNANIMITY: Contabilização por unanimidade

IMPORTANTE para SUBJECT:
- Extrair apenas o objeto da votação (substantivos), NÃO incluir verbos
- Correto: "ata n.º 28 de 20.12.2023", "alterações orçamentais"
- Incorreto: "aprovar a ata n.º 28", "ratificar as alterações orçamentais"

Contexto: Extraia todas as entidades relacionadas com votação em deliberações gerais de câmara municipal,
incluindo votantes, assuntos, e resultados das votações conforme aparecem no texto.
Extraia apenas as entidades mencionadas explicitamente no texto.
"""


def get_examples() -> List[lx.data.ExampleData]:
    """Get Portuguese few-shot examples for voting extraction"""
    
    return [
        # Example 1: Basic voting - Alandroal
        lx.data.ExampleData(
            text="3. ALTERAÇÃO ORÇAMENTAL\nPelo Senhor Presidente foi presente a esta reunião a informação da contabilidade que se anexa à presente ata.\nPonderado e analisado o assunto o Executivo Municipal deliberou por maioria, com os votos a favor dos eleitos pelo PS e a abstenção da eleita pelo Nós, Cidadãos aprovar a alteração orçamental.",
            extractions=[
                lx.data.Extraction(extraction_class="VOTING", extraction_text="deliberou"),
                lx.data.Extraction(extraction_class="COUNTING-MAJORITY", extraction_text="por maioria"),
                lx.data.Extraction(extraction_class="SUBJECT", extraction_text="alteração orçamental"),
                lx.data.Extraction(extraction_class="VOTER-FAVOR", extraction_text="o Executivo Municipal"),
                lx.data.Extraction(extraction_class="VOTER-FAVOR", extraction_text="eleitos pelo PS"),
                lx.data.Extraction(extraction_class="VOTER-ABSTENTION", extraction_text="eleita pelo Nós, Cidadãos")
            ]
        ),
        
        # Example 2: Unanimous voting - Alandroal
        lx.data.ExampleData(
            text="""5. PROCESSOS E REQUERIMENTOS DIVERSOS Pelo Senhor Presidente foi presente a esta reunião o pedido
            de apoio financeiro do Centro Social Paroquial que se anexa à presente ata. Segundo o Sr. Presidente
            trata-se do pedido de apoio que se tem prestado para a sustentabilidade da valência de creche.
            O Centro Social está a funcionar em instalações provisórias, cedidas pelo município,
            e que só permitem ter 15 crianças, o que representa um problema pois há crianças da sede de concelho que não
            têm resposta, não havendo sustentabilidade para o seu funcionamento. A Câmara assumiu com o Centro Social,
            numa altura de pandemia em que as dificuldades eram maiores, que iria tentar compensar esta situação com
            este apoio para que se possa garantir a existência de uma creche na sede de concelho.
            Ponderado e analisado o assunto o Executivo Municipal deliberou por unanimidade, aprovar o pedido de apoio
            financeiro do Centro Social Paroquial para o funcionamento da Creche""",
            extractions=[
                lx.data.Extraction(extraction_class="VOTING", extraction_text="deliberou"),
                lx.data.Extraction(extraction_class="COUNTING-UNANIMITY", extraction_text="por unanimidade"),
                lx.data.Extraction(extraction_class="SUBJECT", extraction_text="pedido de apoio financeiro do Centro Social Paroquial para o funcionamento da Creche"),
                lx.data.Extraction(extraction_class="VOTER-FAVOR", extraction_text="o Executivo Municipal")
            ]
        ),

        # Example 3: Absent voter - Campo Maior
        lx.data.ExampleData(
            text="""
            -Antes de se proceder `deliberação abaixo mencionada, a senhora vereadora Fátima Vitorino, ausentou-se da sala de reuniões por ser parte interessada no assunto.
            -EMISSÃO DE CERTIDÃO AO ABRIGO DO Nº 1 DO ARTIGO 54 DA LEI 91/95, DE 2 DE SETEMBRO, COM A REDAÇÃO
            CONFERIDA PELA LEI 64/2003, DE 23 DE AGOSTO, EM RELAÇÃO AO NEGÓCIO JURÍDICO DOS PRÉDIOS RÚSTICO SITO NO ******, FREGUESIA DE SÃO JOÃO BAPTISTA:
            -EM FACE DA INFORMAÇÃO DA DIVISÃO DE OBRAS E URBANISMO, A CÂMARA DELIBEROU, POR TODOS OS MEMBROS PRESENTES, NOS TERMOS DO NÚMERO 1 DO ARTIGO 54º DA LEI NÚMERO 91/95, DE 02 DE SETEMBRO, COM A REDAÇÃO CONFERIDA PELA LEI 64/2003, DE 23 DE AGOSTO, DAR PARECER FAVORÁVEL AO NEGÓCIO JURÍDICO.
            -Logo após esta votação a senhora vereadora Fátima Vitorino regressou à sala, retomando os trabalhos da reunião.
            """,
            extractions=[
                lx.data.Extraction(extraction_class="VOTER-ABSENT", extraction_text="Fátima Vitorino"),
                lx.data.Extraction(extraction_class="VOTER-FAVOR", extraction_text="A CÂMARA"),
                lx.data.Extraction(extraction_class="VOTING", extraction_text="DELIBEROU"),
                lx.data.Extraction(extraction_class="COUNTING-UNANIMITY", extraction_text="POR TODOS OS MEMBROS PRESENTES"),
                lx.data.Extraction(extraction_class="SUBJECT", extraction_text="PARECER FAVORÁVEL AO NEGÓCIO JURÍDICO"),
            ]
        ),
        
        # Example 4: Against votes - Porto
        lx.data.ExampleData(
            text="""
                13. Aprovação do novo Regulamento Municipal da Feira da Pasteleira.
                A Senhora Vereadora Rosário Gambôa disse que compreendem que não só por questões de reajuste, de espaço e de segurança que tivessem de ser feitas estas alterações à Feira da Pasteleira.
                Aprovada, por maioria, com 1 voto contra do Senhor Vereador do BE e com 4 abstenções dos Senhores Vereadores do PS, da Senhora Vereadora Independente Catarina Santos Cunha e da Senhora Vereadora da CDU.
                """,
            extractions=[
                lx.data.Extraction(extraction_class="SUBJECT", extraction_text="novo Regulamento Municipal da Feira da Pasteleira"),
                lx.data.Extraction(extraction_class="VOTING", extraction_text="Aprovada"),
                lx.data.Extraction(extraction_class="COUNTING-MAJORITY", extraction_text="por maioria"),
                lx.data.Extraction(extraction_class="VOTER-AGAINST", extraction_text="Vereador do BE"),
                lx.data.Extraction(extraction_class="VOTER-ABSTENTION", extraction_text="Vereadores do PS"),
                lx.data.Extraction(extraction_class="VOTER-ABSTENTION", extraction_text="Vereadora Independente Catarina Santos Cunha"),
                lx.data.Extraction(extraction_class="VOTER-ABSTENTION", extraction_text="Vereadora da CDU"),
            ]
        ),
    ]


def get_voting_extraction_config() -> Dict[str, Any]:
    """Get extraction configuration for Gemini 2.0 Flash"""
    
    entity_types = [
        "VOTER-FAVOR", "VOTER-AGAINST", "VOTER-ABSTENTION", "VOTER-ABSENT",
        "SUBJECT", "VOTING", "COUNTING-MAJORITY", "COUNTING-UNANIMITY"
    ]
    
    return {
        "prompt": PROMPT,
        "examples": get_examples(),
        "entity_types": entity_types,
        "language": "pt"
    }
