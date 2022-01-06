import json
import os
import ast

from typing import Optional
from pydantic import BaseModel
from googleapiclient.http import MediaFileUpload
from src.dca_file_uploads.config.settings import get_config
from indigo.common.log import Log
from googleapiclient import discovery
from googleapiclient.discovery import Resource
from google_auth_httplib2 import AuthorizedHttp
from google.oauth2 import service_account
import credstash

LOG = Log.getLogger(__name__)
config = get_config()


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


class DriveFileMetadata(BaseModel):
    id: str
    name: str


class MultipleFoldersFoundException(Exception):
    pass


def make_drive_resource() -> Resource:
    """
    Authenticates and returns a google drive resource.
    """
    google_oauth_creds = ast.literal_eval(
        credstash.getSecret("IA_PIPELINE_GLOBAL_GOOGLE_SHEETS_API_KEY")
    )

    with open("key.json", "w") as fp:
        json.dump(google_oauth_creds, fp)

    creds = service_account.Credentials.from_service_account_file(
        "key.json", scopes=SCOPES
    )
    os.remove("key.json")
    scoped_creds = creds.with_subject(DELEGATE_EMAIL)

    http = AuthorizedHttp(scoped_creds)
    return discovery.build("drive", "v3", http=http)


def find_folder(
    resource: Resource, parent_id: str, folder_name: str
) -> Optional[DriveFileMetadata]:
    """
    Tries to find a folder matching the provided name in the provided parent folder id.

    Raises an exception if multiple matching folders are found.
    """
    kwargs = {
        "q": str(f"parents in '{parent_id}' and name = '{folder_name}' "),
        "corpora": "teamDrive",
        "teamDriveId": config.TEAM_DRIVE_ID,
        "supportsTeamDrives": True,
        "includeTeamDriveItems": True,
        "fields": "files(id, name)",
        "pageSize": 2,
    }
    response = resource.files().list(**kwargs).execute()  # type: ignore
    file_metadatas = [DriveFileMetadata(**r) for r in response["files"]]
    if len(file_metadatas) > 1:
        raise MultipleFoldersFoundException(
            f"Multiple matches for folder titled {folder_name} in parent folder {parent_id}"
        )
    elif len(file_metadatas) == 1:
        return file_metadatas[0]
    else:
        return None


def create_folder(
    resource: Resource, parent_id: str, folder_name: str
) -> DriveFileMetadata:
    """
    Creates a new google drive folder and returns the folder file_id.
    """
    file_metadata = {
        "name": f"{folder_name}",
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }

    kwargs = {
        "supportsTeamDrives": True,
    }

    file = resource.files().create(body=file_metadata, **kwargs).execute()
    return DriveFileMetadata(**file)


def find_or_create_folder(
    resource: Resource, parent_id: str, folder_name: str
) -> DriveFileMetadata:
    folder_metadata = find_folder(resource, parent_id, folder_name)
    if folder_metadata is None:
        folder_metadata = create_folder(resource, parent_id, folder_name)

    return folder_metadata


class MediaTypeException(Exception):
    pass


def move_file(
    resource: Resource, file_id: str, old_parent_id: str, new_parent_id: str
) -> None:
    kwargs = {
        "removeParents": old_parent_id,
        "addParents": new_parent_id,
        "supportsTeamDrives": True,
    }
    resource.files().update(fileId=file_id, **kwargs).execute()


def image_to_google_drive(
    resource: Resource, parent_id: str, local_file_path: str, file_name: str
) -> str:
    """
    Loads image to google drive and returns the file id
    """

    if local_file_path.endswith(".jpeg"):
        mime_type = "image/jpeg"

    elif local_file_path.endswith(".png"):
        mime_type = "image/png"

    else:
        raise MediaTypeException(
            f"Media type for {local_file_path} is not accepted. Accepted types: [.jpeg, .png]"
        )

    media = MediaFileUpload(filename=local_file_path, mimetype=mime_type)
    body = {
        "name": file_name,
        "parents": [parent_id],
    }
    kwargs = {
        "supportsTeamDrives": True,
        "body": body,
        "media_body": media,
    }

    file = resource.files().create(**kwargs).execute()  # type: ignore
    return file["id"]