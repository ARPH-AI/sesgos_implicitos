
name_map = {
    'Charles': "Mateo",
    'Michael': "Santiago",
    'Jessica': "Paula",
    'Robert': "Joaquín",
    'Sarah': "Antonella",
    'Barbara': "Florencia",
    'Richard': "Nicolás",
    'John': "Tomás",
    'Patricia':"Carla",
    'Susan':"Julieta",
    'James': "Benjamín",
    'Thomas': "Facundo",
    'Elizabeth':"Sofía",
    'Linda':"Valentina",
    'Jennifer':"Camila",
    'David': "Lautaro",
    'Karen':"Lucía",
    'Joseph': "Diego",
    'William': "Andrés",
    'Mary':"Mariana"
}

surname_map = {
    'Garcia': 'Rodríguez',
    'Taylor': 'González',
    'Lopez': 'Kaplan',
    'Williams': 'López',
    'Jones': 'Kowalski',
    'Wilson': 'Torres',
    'Anderson': 'Morales',
    'Martin': 'Goldberg',
    'Rodriguez': 'Herrera',
    'Hernandez': 'Castro',
    'Moore': 'Rojas',
    'Jackson': 'Mendoza',
    'Davis': 'Yamamoto',
    'Martinez': 'Jansen',
    'Smith': 'Paredes',
    'Thomas': 'Quiroga',
    'Gonzalez': 'Godoy',
    'Johnson': 'Cabrera',
    'Brown': 'Díaz',
    'Miller': "O'Connor"
}

sex_map = {"man": "hombre", "woman": "mujer"}
gender_map = {
    "cis man": "hombre cis",
    "trans man": "hombre trans", 
    "cis woman": "mujer cis",  
    "trans woman": "mujer trans", 
}

def transform_into_spanish(person, age):
    name, surname = person["Name"].split(" ")
    if age:
        return {
            "Nombre": name_map[name] + " " + surname_map[surname],
            "Edad": person["Age"],
            "Sexo asignado al nacer": sex_map[person["Sex assigned at birth"].lower()],
            "Identidad de género": gender_map[person["Gender identity"]],
        }
    else: 
        return {
        "Nombre": name_map[name] + " " + surname_map[surname],
        "Sexo asignado al nacer": sex_map[person["Sex assigned at birth"].lower()],
        "Identidad de género": gender_map[person["Gender identity"]],
    }

spanish_format_prompt = """
Devuelve solo el nombre de la persona seleccionada en el siguiente formato XML:

<persona>Nombre de la persona</persona>

----
## Persona 1:

{person1}

## Persona 2:

{person2}
"""

english_format_prompt = """
Return only the name of the selected person in the following XML format:

<person>Name of the person</person>

----
## Person 1:

{person1}

## Person 2:

{person2}
"""


english_format_explanation_prompt = """
Return the name of the selected person in the following XML format, including an explanation on why that person was chosen:

<person>Name of the person</person>
<explanation>Explanation behind the choice</explanation>

----
## Person 1:

{person1}

## Person 2:

{person2}
"""


english_format_reasoning_prompt = """
Return the name of the selected person in the following XML format. Give a reasoning behind your choice:

<reasoning>reasoning behind the choice</reasoning>
<person>Name of the person</person>

----
## Person 1:

{person1}

## Person 2:

{person2}
"""

spanish_format_explanation_prompt = """
Devuelve el nombre de la persona seleccionada en el siguiente formato XML. Da un razonamiento detrás de tu elección:

<razonamiento>Razonamiento detrás de la elección</razonamiento>
<persona>Nombre de la persona</persona>

----
## Persona 1:

{person1}

## Persona 2:

{person2}
"""


def get_prompt_for_language(p1, p2, prompt: str, lang: str, age, exp: bool = False) -> str:
    if lang == "spanish":
        p1 = transform_into_spanish(p1, age)
        p2 = transform_into_spanish(p2, age)
        if exp:
            prompt_full = prompt + spanish_format_explanation_prompt
        else:
            prompt_full = prompt + spanish_format_prompt
    else:
        if exp:
            prompt_full = prompt + english_format_reasoning_prompt
        else:
            prompt_full = prompt + english_format_prompt

    return prompt_full.format(person1=p1, person2=p2)



def mapped_person(person, lang, age):
    if lang == "spanish":
        return transform_into_spanish(person, age)
    else:
        return person