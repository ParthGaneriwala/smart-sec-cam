import React from "react";
import io from "socket.io-client";

const SERVER_URL = "http://sec-cam-server:5000"
let socket = io(SERVER_URL)

class ImageViewer extends React.Component {

    constructor(props){
        super(props)
        this.state = {
            srcBlob: null,
            oldUrl: null
        }
    }

    componentDidMount(){
        socket.on('image', (payload) => {
            var image = new Blob( [ new Uint8Array( payload.data ) ], { type: "image/jpeg" } )
            this.setState({
                oldUrl: this.state.srcBlob,
                srcBlob: URL.createObjectURL( image )
            })
            if (this.state.oldUrl != null){
                URL.revokeObjectURL(this.state.oldUrl)
            }
            image = null;
        });
    }

    render() {
        return (
            <div className="imageviewer">
                <img src={this.state.srcBlob} width="50%" height="50%"></img>
            </div>
        );
    }

}

export default ImageViewer