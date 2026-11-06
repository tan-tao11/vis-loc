from records import parse_records


def test_whitespace_key_value_txt(tmp_path):
    f = tmp_path/'xxxx1000.txt'
    f.write_text('number      1000\ndroneLongitude      87.1646099\ndroneLatitude\t42.1075953\n'
                 'droneAltitude    1999.1149902\nsequence   sequence1\n', encoding='utf-8')
    records, fmt = parse_records(f)
    assert fmt == 'kv' and len(records) == 1
    r = records[0]
    assert r['number'] == 1000 and r['droneLongitude'] == 87.1646099 and r['droneLatitude'] == 42.1075953
    assert r['sequence'] == 'sequence1'
