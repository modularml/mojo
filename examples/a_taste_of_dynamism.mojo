#a taste of dynamism in pure mojo
fn main():
    try:

        var db=object([])
        var user = object([])
        
        var field = object(["array"])
        var array = object([123])
        field.append(array)
        
        user.append(field)

        var uuid=object([])
        uuid.append("uuid")
        uuid.append(object(["04F"]))        
        user.append(uuid)
        
        db.append(user)
        
        print(db)
        
        uuid[1]="from array to string"
        array.append(456)
        array[0]=999
        
        print(db)

    except e:
        print(e.value)
