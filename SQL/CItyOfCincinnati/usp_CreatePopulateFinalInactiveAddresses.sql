
CREATE PROCEDURE [dbo].[usp_CreatePopulateFinalInactiveAddresses]
AS

BEGIN
BEGIN TRY

  IF OBJECT_ID('tmp.FinalInactiveAddresses','U') IS NOT NULL
  DROP TABLE tmp.FinalInactiveAddresses 

 CREATE TABLE [tmp].[FinalInactiveAddresses](
	[HOUSENUMBER] [float] NULL,
	[DIRECTION] [nvarchar](255) NULL,
	[STREETNAME] [nvarchar](255) NULL,
	[SUFFIX] [nvarchar](255) NULL,
	[Unit#] [nvarchar](255) NULL,
	[CITY] [nvarchar](255) NULL,
	[STATE] [nvarchar](255) NULL,
	[ZIPCODE] [float] NULL,
	[ADDRESS] [nvarchar](255) NULL,
	[FULLADDRESS] [nvarchar](255) NULL,
	[SNA_NEIGHBORHOOD] [nvarchar](255) NULL,
	[STRUCTURE_TYPE] [nvarchar](255) NULL,
	[ROUTE] [nvarchar](255) NULL,
	[DAY] [nvarchar](255) NULL,
	[ROUTE1] [nvarchar](255) NULL,
	[COLOR] [nvarchar](255) NULL,
	[CART -Y/N] [nvarchar](255) NULL
) ON [PRIMARY]

  -- creating a refined table by excluding some null values
  INSERT INTO [tmp].[FinalInactiveAddresses]
   SELECT  [HOUSENUMBER]
      ,[DIRECTION]
      ,[STREETNAME]
      ,[SUFFIX]
      ,[Unit#]
      ,[CITY]
      ,[STATE]
      ,[ZIPCODE]
	  ,REPLACE(convert(varchar,ISNULL([HOUSENUMBER],''))+ISNULL([DIRECTION],'')+ISNULL([STREETNAME],'')+ISNULL([SUFFIX],''),' ','') as ADDRESS
      ,[FULLADDRESS]
      ,[SNA_NEIGHBORHOOD]
      ,[STRUCTURE_TYPE]
      ,[ROUTE]
      ,[DAY]
      ,'GOLD' as [ROUTE1]
      ,[COLOR]
      ,[CART -Y/N]
  FROM [City of Cincinnati].[dbo].[FinalInactiveGold]
  UNION
  SELECT  [HOUSENUMBER]
      ,[DIRECTION]
      ,[STREETNAME]
      ,[SUFFIX]
      ,[Unit#]
      ,[CITY]
      ,[STATE]
      ,[ZIPCODE]
	  ,REPLACE(convert(varchar,ISNULL([HOUSENUMBER],''))+ISNULL([DIRECTION],'')+ISNULL([STREETNAME],'')+ISNULL([SUFFIX],''),' ','') as ADDRESS
      ,[FULLADDRESS]
      ,[SNA_NEIGHBORHOOD]
      ,[STRUCTURE_TYPE]
      ,[ROUTE]
      ,[DAY]
      ,'GREEN' as [ROUTE1]
      ,[COLOR]
      ,[CART -Y/N]
  FROM [City of Cincinnati].[dbo].[FinalInactiveGreen]

	 BEGIN TRANSACTION 
	 IF EXISTS( Select 1 FROM [tmp].[FinalInactiveAddresses])
	 BEGIN
	 IF OBJECT_ID('dbo.FinalInactiveAddresses','U') IS NOT NULL
	 BEGIN
	 DROP TABLE [dbo].[FinalInactiveAddresses]
	 END
	 ALTER SCHEMA dbo TRANSFER [tmp].[FinalInactiveAddresses]
	 END
	 COMMIT TRANSACTION 

	
 
END TRY

BEGIN CATCH 
  IF (@@TRANCOUNT > 0)
   BEGIN
      ROLLBACK TRANSACTION 
      PRINT 'Error detected, all changes reversed'
   END 
    SELECT
        ERROR_NUMBER() AS ErrorNumber,
        ERROR_SEVERITY() AS ErrorSeverity,
        ERROR_STATE() AS ErrorState,
        ERROR_PROCEDURE() AS ErrorProcedure,
        ERROR_LINE() AS ErrorLine,
        ERROR_MESSAGE() AS ErrorMessage
END CATCH
END