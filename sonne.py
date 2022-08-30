import os,re, tempfile, xml.etree.ElementTree as ET
import statistics

def evaluate(directory_xml):

    path_to_sonne=('C:/Program Files (x86)/AMPS/ECGSolve')   #path to Sonne

    files_result=[]
    all_annotations=[]
    list_of_files = os.listdir(directory_xml)
    files = [x for x in list_of_files if x.endswith(".xml")]
    for file in files:
        path_to_xml = str(directory_xml + str(file))
        path_out_sonne="C:/Users/simon/Desktop/Tesi/pyProject/sonne_results/" + file + '.xml'
        stream=os.popen('cd '+ path_to_sonne + '&& ' + 'ECGSolve -i ' + path_to_xml + ' -m 4 -o' + path_out_sonne)

        output=stream.read() # print Solve Work
        print(output)

        #generate root for xml file
        tree = ET.parse(path_out_sonne)
        root = tree.getroot()
        # going through tree searching for data and storing in dict
        annotations_dict = {}
        for components in root.findall('./{urn:hl7-org:v3}component/{urn:hl7-org:v3}series/{urn:hl7-org:v3}subjectOf/{urn:hl7-org:v3}annotationSet/'):
            for component in components.findall('./{urn:hl7-org:v3}annotation'):
                # creating a dictionary with names of waves annotation, one dict for file
                for name, values in zip(component.findall('./{urn:hl7-org:v3}code'),
                                        component.findall('./{urn:hl7-org:v3}value')):
                    code = (name.get(key='code'))
                    value = (values.get(key='value'))
                    # PP names
                    if re.findall('_PP$', code):
                        if 'PP' not in annotations_dict.keys():
                            annotations_dict['PP'] = []
                        annotations_dict['PP'].append(int(value))
                    # RR Names
                    if re.findall('_RR$', code):
                        if 'RR' not in annotations_dict.keys():
                            annotations_dict['RR'] = []
                        annotations_dict['RR'].append(int(value))
                    # PR names
                    if re.findall('_PR$', code):
                        if 'PR' not in annotations_dict.keys():
                            annotations_dict['PR'] = []
                        annotations_dict['PR'].append(int(value))
                    # QRS Names
                    if re.findall('_QRS$', code):
                        if 'QRS' not in annotations_dict.keys():
                            annotations_dict['QRS'] = []
                        annotations_dict['QRS'].append(int(value))
                    # QT Names
                    if re.findall('_QT$', code):
                        if 'QT' not in annotations_dict.keys():
                            annotations_dict['QT'] = []
                        annotations_dict['QT'].append(int(value))
                    # JT Names
                    if re.findall('_JT$', code):
                        if 'JT' not in annotations_dict.keys():
                            annotations_dict['JT'] = []
                        annotations_dict['JT'].append(int(value))
                    # ST Names
                    if re.findall('_ST$', code):
                        if 'ST' not in annotations_dict.keys():
                            annotations_dict['ST'] = []
                        annotations_dict['ST'].append(int(value))
                    # AMPL_T_MAX
                    if re.findall('AMPL_T_MAX$', code):
                        if 'AMPL_T_MAX' not in annotations_dict.keys():
                            annotations_dict['AMPL_T_MAX'] = []
                        annotations_dict['AMPL_T_MAX'].append(int(value))
                    # AMPL_R
                    if re.findall('AMPL_R$', code):
                        if 'AMPL_R' not in annotations_dict.keys():
                            annotations_dict['AMPL_R'] = []
                        annotations_dict['AMPL_R'].append(int(value))

        all_annotations.append(annotations_dict)
        files_result.append({file: all_annotations})
    return files_result



