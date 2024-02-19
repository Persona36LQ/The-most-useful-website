import {useState} from "react";

export default function TopPanel() {

    const [show, setShow] = useState(false)

    return (
        <div className="TopPanel"
             onMouseEnter={setShow(true)}
             onMouseLeave={setShow(false)}
             style={{
            height: show ? "250px" : "25px"
        }}>

        </div>
    )
}